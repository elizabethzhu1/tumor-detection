"""
Download public TCGA masked somatic mutation MAFs from the GDC API.

The downloader queries the GDC files endpoint for open-access MAF files from
the Aliquot Ensemble Somatic Variant Merging and Masking workflow, then
downloads each matching file through the GDC data endpoint.
"""

import json
import os
import shutil
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import certifi
except ImportError:
    certifi = None


GDC_API_BASE = "https://api.gdc.cancer.gov"
DEFAULT_TCGA_PROJECTS = {
    "TCGA-SKCM": "SKCM",
    "TCGA-LUAD": "LUAD",
    "TCGA-BRCA": "BRCA",
    "TCGA-UCEC": "UCEC",
    "TCGA-COAD": "COAD",
}
DEFAULT_WORKFLOW = "Aliquot Ensemble Somatic Variant Merging and Masking"


def _ssl_context():
    if certifi is None:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def parse_project_specs(specs):
    """
    Parse CLI project specs.

    Accepted forms:
      TCGA-BRCA
      TCGA-BRCA:BRCA

    Returns:
      dict mapping GDC project id -> local tumor label.
    """
    if not specs:
        return dict(DEFAULT_TCGA_PROJECTS)

    out = {}
    for spec in specs:
        if ":" in spec:
            project_id, label = spec.split(":", 1)
        else:
            project_id = spec
            label = spec.replace("TCGA-", "")
        project_id = project_id.strip()
        label = label.strip()
        if not project_id or not label:
            raise ValueError(f"Invalid TCGA project spec: {spec!r}")
        out[project_id] = label
    return out


def _post_json(url, payload, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GDC API request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"GDC API request failed: {exc}") from exc


def query_maf_files(project_id, workflow_type=DEFAULT_WORKFLOW, page_size=1000,
                    max_files=None):
    """
    Find public masked somatic mutation MAF files for a TCGA project.

    Returns:
      list of file metadata dicts from the GDC files endpoint.
    """
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_id,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_type",
                    "value": "Masked Somatic Mutation",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.data_format",
                    "value": "MAF",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": workflow_type,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "files.access",
                    "value": "open",
                },
            },
        ],
    }
    fields = [
        "file_id",
        "file_name",
        "md5sum",
        "file_size",
        "data_format",
        "data_type",
        "access",
        "analysis.workflow_type",
        "cases.project.project_id",
    ]
    hits = []
    offset = 0
    while True:
        remaining = None if max_files is None else max_files - len(hits)
        if remaining is not None and remaining <= 0:
            break
        current_size = page_size if remaining is None else min(page_size, remaining)
        payload = {
            "filters": filters,
            "format": "JSON",
            "fields": ",".join(fields),
            "from": str(offset),
            "size": str(current_size),
            "sort": "file_name:asc",
        }
        data = _post_json(f"{GDC_API_BASE}/files", payload)
        page_hits = data.get("data", {}).get("hits", [])
        hits.extend(page_hits)

        pagination = data.get("data", {}).get("pagination", {})
        total = int(pagination.get("total", len(hits)))
        if not page_hits or len(hits) >= total:
            break
        offset += len(page_hits)
    return hits


def _download_file(file_id, destination, force=False, timeout=120, retries=3,
                   retry_sleep=2.0):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    url = f"{GDC_API_BASE}/data/{file_id}"
    req = Request(url, method="GET")
    retryable_http_codes = {500, 502, 503, 504}
    for attempt in range(retries + 1):
        try:
            with (
                urlopen(req, timeout=timeout, context=_ssl_context()) as resp,
                open(tmp_path, "wb") as out,
            ):
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            os.replace(tmp_path, destination)
            return True
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in retryable_http_codes and attempt < retries
            if not should_retry:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(
                    f"Download failed for {file_id} with HTTP {exc.code}: {detail}"
                ) from exc
            wait = retry_sleep * (2 ** attempt)
            print(
                f"    retrying {file_id} after HTTP {exc.code} "
                f"({attempt + 1}/{retries})..."
            )
            time.sleep(wait)
        except URLError as exc:
            if attempt >= retries:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(f"Download failed for {file_id}: {exc}") from exc
            wait = retry_sleep * (2 ** attempt)
            print(
                f"    retrying {file_id} after network error "
                f"({attempt + 1}/{retries})..."
            )
            time.sleep(wait)

    return False


def download_tcga_mafs(projects, output_dir, force=False, workflow_type=DEFAULT_WORKFLOW,
                       max_files_per_project=None, retries=3,
                       skip_failed_downloads=False):
    """
    Download TCGA MAF files and return a mapping of tumor label -> file paths.

    Args:
      projects: dict mapping project id, e.g. TCGA-BRCA, to tumor label, e.g. BRCA.
      output_dir: directory where files are stored under one subdirectory per label.
      force: redownload even if the destination file already exists.
    """
    output_dir = Path(output_dir)
    maf_paths = {}

    for project_id, label in projects.items():
        print(f"  querying GDC for {project_id} ({label})...")
        hits = query_maf_files(
            project_id,
            workflow_type=workflow_type,
            max_files=max_files_per_project,
        )
        if not hits:
            raise RuntimeError(
                f"No public MAF files found for {project_id} using workflow "
                f"{workflow_type!r}."
            )

        label_paths = []
        for hit in hits:
            file_id = hit.get("file_id") or hit.get("id")
            file_name = hit.get("file_name") or f"{file_id}.maf.gz"
            if not file_id:
                raise RuntimeError(
                    f"GDC file hit for {project_id} lacked file_id: {hit}"
                )
            destination = output_dir / label / file_name
            try:
                downloaded = _download_file(
                    file_id,
                    destination,
                    force=force,
                    retries=retries,
                )
                action = "downloaded" if downloaded else "cached"
                print(f"    {action}: {destination}")
                label_paths.append(str(destination))
            except RuntimeError as exc:
                if not skip_failed_downloads:
                    raise
                print(f"    skipped failed download: {file_id} ({exc})")
        if not label_paths:
            raise RuntimeError(f"All downloads failed for {project_id} ({label}).")
        maf_paths[label] = label_paths

    return maf_paths

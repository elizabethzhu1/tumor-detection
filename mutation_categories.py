"""
Defines the canonical 96 mutation categories used in COSMIC SBS analysis.

A mutation is categorized by:
  - the substitution type (6 pyrimidine-centric types: C>A, C>G, C>T, T>A, T>C, T>G)
  - the 5' base (A, C, G, T)
  - the 3' base (A, C, G, T)
giving 6 * 4 * 4 = 96 categories.

By convention, mutations are reported relative to the pyrimidine strand: if the
reference base is a purine (A or G), we take the reverse complement of the
trinucleotide and the substitution so it's expressed as C>X or T>X. This is
because DNA is double-stranded and (e.g.) a G>A mutation on one strand is a
C>T on the other.
"""

from itertools import product

PYRIMIDINES = ["C", "T"]
PURINES = ["A", "G"]
BASES = ["A", "C", "G", "T"]
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

# The six pyrimidine-referenced substitution types.
SUBSTITUTIONS = [
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]


def build_category_list():
    """Return the 96 categories as strings like 'A[C>A]A', in canonical order."""
    cats = []
    for ref, alt in SUBSTITUTIONS:
        for five_prime in BASES:
            for three_prime in BASES:
                cats.append(f"{five_prime}[{ref}>{alt}]{three_prime}")
    return cats


CATEGORIES = build_category_list()
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
assert len(CATEGORIES) == 96


def reverse_complement(seq):
    return "".join(COMPLEMENT[b] for b in reversed(seq))


def classify_mutation(ref, alt, trinuc_context):
    """
    Map a (ref, alt, 5'-N-3' trinucleotide) to one of the 96 category indices.

    Args:
        ref: reference base, single character
        alt: alternate base, single character
        trinuc_context: 3-character string, the 5' base, ref, 3' base on the
            reference strand
    Returns:
        integer index 0..95, or None if the mutation is not a valid SBS
        (e.g., contains N, indel, or ref doesn't match context middle base).
    """
    if trinuc_context is None:
        return None
    if len(trinuc_context) != 3 or "N" in trinuc_context:
        return None
    if ref not in BASES or alt not in BASES or ref == alt:
        return None
    if trinuc_context[1] != ref:
        return None

    # Fold to pyrimidine reference.
    if ref in PURINES:
        ref = COMPLEMENT[ref]
        alt = COMPLEMENT[alt]
        trinuc_context = reverse_complement(trinuc_context)

    key = f"{trinuc_context[0]}[{ref}>{alt}]{trinuc_context[2]}"
    return CATEGORY_TO_IDX.get(key)


if __name__ == "__main__":
    # Sanity checks.
    print(f"Built {len(CATEGORIES)} categories.")
    print("First 5:", CATEGORIES[:5])
    print("Last 5:", CATEGORIES[-5:])

    # C>T at A_A context: should be A[C>T]A
    idx = classify_mutation("C", "T", "ACA")
    assert CATEGORIES[idx] == "A[C>T]A", CATEGORIES[idx]

    # G>A at A_A on reference -> reverse complement -> T[C>T]T
    idx = classify_mutation("G", "A", "AGA")
    assert CATEGORIES[idx] == "T[C>T]T", CATEGORIES[idx]

    print("All sanity checks passed.")

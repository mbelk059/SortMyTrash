"""
Illustrative class to bin labels. Municipal recycling rules vary by city and program.
"""

DISCLAIMER = "Local rules differ; this mapping is illustrative only, not disposal advice."

def suggested_bin(class_name: str) -> str:
    n = (class_name or "").lower().strip()
    if n in ("plastic", "metal", "paper", "cardboard", "glass"):
        return "Blue — recycling (illustrative)"
    if n == "organic":
        return "Green — organics (illustrative)"
    if n == "trash":
        return "Black — garbage (illustrative)"
    return "Unknown"


def bin_mapping_lines():
    return [
        "Illustrative bin mapping (rules vary by municipality):",
        "  plastic, metal, paper, cardboard, glass → Blue / recycling",
        "  organic → Green / organics",
        "  trash → Black / garbage",
        f"  ({DISCLAIMER})",
    ]


def print_bin_mapping_reference():
    print("\n" + "\n".join(bin_mapping_lines()) + "\n")

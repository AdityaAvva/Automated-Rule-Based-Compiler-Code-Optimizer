"""Pre-optimization source and original-TAC features for benchmark ML models."""

import re
from collections import Counter

FEATURE_NAMES = [
    "source_characters",
    "source_lines",
    "assignment_count",
    "print_count",
    "numeric_literal_count",
    "identifier_occurrences",
    "unique_identifiers",
    "variable_reads",
    "unique_variables_written",
    "addition_count",
    "subtraction_count",
    "multiplication_count",
    "division_count",
    "modulo_count",
    "constant_binary_expression_count",
    "copy_assignment_count",
    "repeated_expression_count",
    "add_zero_identity_count",
    "subtract_zero_identity_count",
    "multiply_one_identity_count",
    "divide_one_identity_count",
    "multiply_zero_identity_count",
    "multiply_two_count",
    "unused_assignment_candidate_count",
    "loop_count",
    "loop_body_statement_count",
    "estimated_loop_iterations",
    "original_tac_instruction_count",
    "original_tac_operation_count",
    "original_tac_temporary_count",
]

IDENTIFIER = r"[A-Za-z_]\w*"
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
ASSIGNMENT = re.compile(rf"\b({IDENTIFIER})\s*=\s*([^;{{}}]+)\s*;")
BINARY_EXPRESSION = re.compile(
    rf"({IDENTIFIER}|{NUMBER})\s*([+\-*/%])\s*({IDENTIFIER}|{NUMBER})"
)
LOOP = re.compile(
    rf"for\s*\(\s*({IDENTIFIER})\s*=\s*(-?\d+)\s*;\s*\1\s*<\s*(\d+)\s*;"
    rf"\s*\1\s*=\s*\1\s*\+\s*1\s*\)\s*\{{([^{{}}]*)\}}"
)


def extract_features(source, original_tac=""):
    """Return fixed-order numeric features computed before optimization."""
    text = source.lower()
    assignments = list(ASSIGNMENT.finditer(text))
    written_names = [match.group(1) for match in assignments]
    identifier_counts = Counter(
        name for name in re.findall(rf"\b{IDENTIFIER}\b", text)
        if name not in {"print", "for"}
    )

    expressions = []
    operator_counts = Counter()
    for match in BINARY_EXPRESSION.finditer(text):
        left, operator, right = match.groups()
        operator_counts[operator] += 1
        if left.isalpha() or "_" in left:
            expressions.append((operator, left, right))
        else:
            expressions.append((operator, left, right))
    normalized_expressions = []
    for operator, left, right in expressions:
        operands = tuple(sorted((left, right))) if operator in {"+", "*"} else (left, right)
        normalized_expressions.append((operator, *operands))
    expression_counts = Counter(normalized_expressions)
    repeated_expression_count = sum(count - 1 for count in expression_counts.values() if count > 1)
    constant_binary_count = sum(
        bool(re.fullmatch(NUMBER, left)) and bool(re.fullmatch(NUMBER, right))
        for _, left, right in expressions
    )

    lower_assignments = [match.group(2).strip() for match in assignments]
    assignment_counts = Counter(written_names)
    unused_assignment_count = sum(
        count for name, count in assignment_counts.items()
        if identifier_counts[name] <= count
    )
    identity_patterns = {
        "add_zero_identity_count": rf"(?:{IDENTIFIER}\s*\+\s*0\b|0\s*\+\s*{IDENTIFIER})",
        "subtract_zero_identity_count": rf"{IDENTIFIER}\s*-\s*0\b",
        "multiply_one_identity_count": rf"(?:{IDENTIFIER}\s*\*\s*1\b|1\s*\*\s*{IDENTIFIER})",
        "divide_one_identity_count": rf"{IDENTIFIER}\s*/\s*1\b",
        "multiply_zero_identity_count": rf"(?:{IDENTIFIER}\s*\*\s*0\b|0\s*\*\s*{IDENTIFIER})",
    }
    loop_matches = list(LOOP.finditer(text))
    loop_count = len(loop_matches)
    loop_body_statement_count = sum(
        len([part for part in match.group(4).split(";") if part.strip()])
        for match in loop_matches
    )
    estimated_loop_iterations = sum(
        max(0, min(int(match.group(3)), int(match.group(2)) + 1000) - int(match.group(2)))
        for match in loop_matches
    )
    tac_lines = [line.strip() for line in original_tac.splitlines() if line.strip()]
    tac_operation_count = sum(
        bool(re.search(r" = .+ [+\-*/%] .+", line))
        for line in tac_lines
    )
    tac_temporary_count = len({
        name for line in tac_lines
        for name in re.findall(r"\bt\d+\b", line)
    })
    values = [
        len(source),
        len(source.splitlines()),
        len(assignments),
        len(re.findall(r"\bprint\b", text)),
        len(re.findall(rf"\b{NUMBER}\b", text)),
        len(re.findall(rf"\b{IDENTIFIER}\b", text)),
        len(set(re.findall(rf"\b{IDENTIFIER}\b", text)) - {"print", "for"}),
        max(0, sum(identifier_counts.values()) - len(written_names)),
        len(set(written_names)),
        operator_counts["+"],
        operator_counts["-"],
        operator_counts["*"],
        operator_counts["/"],
        operator_counts["%"],
        constant_binary_count,
        sum(bool(re.fullmatch(IDENTIFIER, rhs)) for rhs in lower_assignments),
        repeated_expression_count,
        *(len(re.findall(pattern, text)) for pattern in identity_patterns.values()),
        len(re.findall(rf"(?:{IDENTIFIER}\s*\*\s*2\b|2\s*\*\s*{IDENTIFIER})", text)),
        unused_assignment_count,
        loop_count,
        loop_body_statement_count,
        estimated_loop_iterations,
        len(tac_lines),
        tac_operation_count,
        tac_temporary_count,
    ]
    return [float(value) for value in values]

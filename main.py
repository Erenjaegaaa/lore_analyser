"""
main.py — CLI entry point for the GraphRAG Lore Assistant.

Usage:
    python main.py query "Who forged the One Ring?"
    python main.py query "What is the relationship between Aragorn and Isildur?"
    python main.py interactive
"""

import sys
from pipeline.query_pipeline import query, print_result
from utils.logger import get_logger

log = get_logger(__name__)


def run_query(question: str) -> None:
    """Run a single query and print the result."""
    result = query(question)
    print_result(result)


def run_interactive() -> None:
    """Run an interactive question-answer loop."""
    print("\n🧙 GraphRAG Lore Assistant — Middle-earth Q&A")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = query(question)
        print_result(result)


def main() -> None:
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python main.py query \"Your question here\"")
        print("  python main.py interactive")
        sys.exit(1)

    mode = args[0].lower()

    if mode == "query":
        if len(args) < 2:
            print("Please provide a question.")
            print("  python main.py query \"Who forged the One Ring?\"")
            sys.exit(1)
        question = " ".join(args[1:])
        run_query(question)

    elif mode == "interactive":
        run_interactive()

    else:
        print(f"Unknown mode: {mode}")
        print("Use 'query' or 'interactive'")
        sys.exit(1)


if __name__ == "__main__":
    main()
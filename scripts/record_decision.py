import argparse
import datetime
import os
import sys

def record_decision(decision_id, decision_text, rationale, status="Approved"):
    log_path = "docs/DECISION_LOG.md"
    date_str = datetime.date.today().isoformat()
    
    new_entry = f"| {decision_id} | {date_str} | {decision_text} | {rationale} | {status} |\n"
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Find the table and insert after the header
    table_start = -1
    for i, line in enumerate(lines):
        if "| ID | Date | Decision | Rationale | Status |" in line:
            table_start = i + 2 # Skip header and separator
            break
    
    if table_start != -1:
        lines.insert(table_start, new_entry)
    else:
        lines.append(new_entry)

    with open(log_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Decision {decision_id} recorded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a design decision.")
    parser.add_argument("--id", required=True, help="Decision ID (e.g., D-005)")
    parser.add_argument("--text", required=True, help="The decision made")
    parser.add_argument("--rationale", required=True, help="Why the decision was made")
    parser.add_argument("--status", default="Approved", help="Status of the decision")
    
    args = parser.parse_args()
    record_decision(args.id, args.text, args.rationale, args.status)

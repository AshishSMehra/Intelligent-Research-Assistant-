#!/usr/bin/env python3
"""
Script to fix missing imports in Python files.
"""

import hashlib
import json
import os
import random
import re
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from botocore.exceptions import NoCredentialsError
from datasets import Dataset, DatasetDict
from flask import request
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, pipeline


def add_missing_imports(file_path):
    """Add missing imports to a file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Common imports that are missing
    missing_imports = {
        "os": "import os",
        "time": "import time",
        "json": "import json",
        "hashlib": "import hashlib",
        "numpy": "import numpy as np",
        "pandas": "import pandas as pd",
        "dataclasses": "from dataclasses import dataclass, asdict, field",
        "typing": "from typing import List, Dict, Any, Optional, Tuple, Set, Union",
        "collections": "from collections import deque",
        "datetime": "from datetime import datetime",
        "random": "import random",
        "re": "import re",
        "uuid": "import uuid",
        "requests": "import requests",
        "flask": "from flask import request",
        "sklearn": "from sklearn.metrics import accuracy_score, precision_recall_fscore_support",
        "transformers": "from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, pipeline",
        "datasets": "from datasets import Dataset, DatasetDict",
        "botocore": "from botocore.exceptions import NoCredentialsError",
    }

    # Check which imports are missing
    lines = content.split("\n")
    existing_imports = set()

    for line in lines:
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            for module in missing_imports:
                if module in line:
                    existing_imports.add(module)

    # Add missing imports
    new_imports = []
    for module, import_stmt in missing_imports.items():
        if module not in existing_imports:
            new_imports.append(import_stmt)

    if new_imports:
        # Find the first import line
        import_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                import_index = i
                break

        if import_index == -1:
            # No imports found, add after docstring
            for i, line in enumerate(lines):
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    import_index = i + 1
                    break
                elif line.strip() and not line.strip().startswith("#"):
                    import_index = i
                    break

        # Insert new imports
        for import_stmt in reversed(new_imports):
            lines.insert(import_index, import_stmt)

        # Write back the file
        with open(file_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Added {len(new_imports)} imports to {file_path}")


def main():
    """Main function to fix imports."""
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        if ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    print(f"Found {len(python_files)} Python files")

    # Fix each file
    for file_path in python_files:
        try:
            add_missing_imports(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Import fixes completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MycoExtract 项目完整性验证脚本

检查所有必要文件是否存在，确保项目可以正常运行
"""

import sys
from pathlib import Path

# 定义必要的文件和目录
REQUIRED_FILES = {
    "根目录": [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        ".gitignore",
    ],
    "配置文件": [
        "config/extraction_config.yaml",
        "config/extraction_config_v7_expanded.yaml",
    ],
    "提示词": [
        "prompts/prompts_extract_from_text.txt",
        "prompts/prompts_extract_from_table.txt",
        "prompts/prompts_extract_from_text_v7_expanded.txt",
        "prompts/prompts_extract_from_table_v7_expanded.txt",
    ],
    "脚本": [
        "scripts/run_extraction.py",
    ],
    "文档": [
        "docs/ARCHITECTURE.md",
        "docs/EXTRACTION_FLOW.md",
        "docs/API_REFERENCE.md",
    ],
    "示例": [
        "examples/basic_usage.py",
    ],
}

REQUIRED_DIRS = {
    "src": ["agents", "extractors", "llm_clients", "llm_extraction", "pipeline", "utils"],
}

REQUIRED_PY_FILES = {
    "src/agents": ["aggregation_agent.py"],
    "src/extractors": ["paper_level_extractor.py"],
    "src/pipeline": ["enhanced_pipeline.py", "paper_level_prechecker.py"],
    "src/utils": ["data_validator.py", "quality_analyzer.py", "unit_normalizer.py"],
}


def check_file_exists(file_path: Path, category: str) -> bool:
    """检查文件是否存在"""
    if file_path.exists():
        print(f"  [OK] {file_path}")
        return True
    else:
        print(f"  [MISSING] {file_path}")
        return False


def check_dir_exists(dir_path: Path, category: str) -> bool:
    """检查目录是否存在"""
    if dir_path.exists() and dir_path.is_dir():
        print(f"  [OK] {dir_path}/")
        return True
    else:
        print(f"  [MISSING] {dir_path}/")
        return False


def validate_project(project_root: Path):
    """验证项目完整性"""
    print("=" * 60)
    print("MycoExtract 项目完整性验证")
    print("=" * 60)

    all_checks_passed = True

    # 检查文件
    for category, files in REQUIRED_FILES.items():
        print(f"\n{category}:")
        for file in files:
            file_path = project_root / file
            if not check_file_exists(file_path, category):
                all_checks_passed = False

    # 检查目录
    print(f"\n目录结构:")
    for dir_name in REQUIRED_DIRS.keys():
        dir_path = project_root / dir_name
        if not check_dir_exists(dir_path, dir_name):
            all_checks_passed = False

    # 检查核心Python文件
    print(f"\n核心模块:")
    for dir_name, files in REQUIRED_PY_FILES.items():
        dir_path = project_root / dir_name
        for file in files:
            file_path = dir_path / file
            if not check_file_exists(file_path, dir_name):
                all_checks_passed = False

    # 检查__init__.py文件
    print(f"\nPython包初始化:")
    for dir_name in REQUIRED_DIRS["src"]:
        init_file = project_root / "src" / dir_name / "__init__.py"
        if not check_file_exists(init_file, dir_name):
            all_checks_passed = False

    # 验证requirements.txt
    print(f"\n依赖检查:")
    req_file = project_root / "requirements.txt"
    if req_file.exists():
        with open(req_file, 'r', encoding='utf-8') as f:
            requirements = f.read()
        required_packages = ["pyyaml", "requests", "python-dotenv"]
        missing_packages = []
        for pkg in required_packages:
            if pkg.lower() not in requirements.lower():
                missing_packages.append(pkg)
        if missing_packages:
            print(f"  [WARNING] requirements.txt 缺少以下包: {missing_packages}")
            all_checks_passed = False
        else:
            print(f"  [OK] requirements.txt 包含必要依赖")
    else:
        print(f"  [MISSING] requirements.txt")
        all_checks_passed = False

    # 最终结果
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("[SUCCESS] 所有检查通过！项目结构完整。")
        print("\n您可以运行:")
        print("  pip install -r requirements.txt")
        print("  python scripts/run_extraction.py --help")
        return 0
    else:
        print("[FAILED] 检查失败！请确保所有必要文件都存在。")
        return 1


def main():
    # 获取项目根目录
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    print(f"项目根目录: {project_root}")
    print()

    # 验证项目
    return validate_project(project_root)


if __name__ == "__main__":
    sys.exit(main())

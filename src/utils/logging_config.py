"""
日志配置模块 - 统一配置项目日志

所有日志文件保存在 logs/ 目录下
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_filename: str = None,
    module_name: str = None
) -> logging.Logger:
    """
    配置项目日志
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
        log_to_file: 是否保存到文件
        log_filename: 日志文件名（None则自动生成）
        module_name: 模块名称（用于logger名称）
        
    Returns:
        配置好的 logger
    """
    # 确保 logs 目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成日志文件名
    if log_to_file and not log_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        module_prefix = f"{module_name}_" if module_name else ""
        log_filename = f"{module_prefix}extraction_{timestamp}.log"
    
    # 配置格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置处理器
    handlers = [logging.StreamHandler()]  # 控制台输出
    
    if log_to_file and log_filename:
        log_file_path = log_dir / log_filename
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # 强制重新配置
    )
    
    # 返回模块专用logger
    logger = logging.getLogger(module_name or __name__)
    
    if log_to_file and log_filename:
        logger.info(f"📝 Logging to: {log_file_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取命名logger
    
    Args:
        name: Logger名称
        
    Returns:
        Logger实例
    """
    return logging.getLogger(name)

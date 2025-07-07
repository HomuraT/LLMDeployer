import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
from loguru import logger


class LoguruManager:
    """
    统一的Loguru日志管理器
    
    提供以下功能：
    1. 主日志文件 - 记录所有程序日志
    2. 模型特定日志文件 - 为每个模型创建独立日志
    3. 动态日志级别调整
    4. 日志轮转配置
    5. 线程安全的日志操作
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式确保全局只有一个日志管理器实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._current_console_level = "INFO"
        self._model_loggers: Dict[str, int] = {}  # 存储模型特定的logger ID
        self._console_handler_id: Optional[int] = None
        self._main_handler_id: Optional[int] = None
        
        # 创建日志目录
        self.log_base_dir = Path("logs")
        self.model_log_dir = Path("vllm_logs")
        self.log_base_dir.mkdir(exist_ok=True)
        self.model_log_dir.mkdir(exist_ok=True)
        
        # 初始化日志配置
        self._setup_main_logging()
    
    def _setup_main_logging(self) -> None:
        """
        设置主要的日志配置
        
        :return: None
        :rtype: None
        """
        # 移除默认的控制台处理器
        logger.remove()
        
        # 详细的日志格式
        detailed_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 简化的控制台格式
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 添加控制台处理器
        self._console_handler_id = logger.add(
            sys.stderr,
            format=console_format,
            level=self._current_console_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 添加主日志文件处理器（带轮转）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_file = self.log_base_dir / f"main_{timestamp}.log"
        
        self._main_handler_id = logger.add(
            main_log_file,
            format=detailed_format,
            level="DEBUG",  # 文件记录所有级别
            rotation="100 MB",  # 按大小轮转
            retention="30 days",  # 保留30天
            compression="zip",  # 压缩旧日志
            backtrace=True,
            diagnose=True,
            enqueue=True  # 线程安全
        )
        
        # 添加按时间轮转的日志文件
        daily_log_file = self.log_base_dir / "daily.log"
        logger.add(
            daily_log_file,
            format=detailed_format,
            level="INFO",
            rotation="00:00",  # 每天轮转
            retention="7 days",  # 保留7天
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        logger.info("Loguru日志系统初始化完成")
        logger.info(f"主日志文件: {main_log_file}")
        logger.info(f"日志目录: {self.log_base_dir.absolute()}")
    
    def set_console_level(self, level: str) -> None:
        """
        动态调整控制台日志级别
        
        :param level: 日志级别 ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        :type level: str
        :return: None
        :rtype: None
        """
        if self._console_handler_id is not None:
            logger.remove(self._console_handler_id)
        
        self._current_console_level = level.upper()
        
        console_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <5}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        )
        
        self._console_handler_id = logger.add(
            sys.stderr,
            format=console_format,
            level=self._current_console_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        logger.info(f"控制台日志级别已调整为: {self._current_console_level}")
    
    def get_model_logger(self, model_name: str) -> 'logger':
        """
        为特定模型获取专用的logger
        
        :param model_name: 模型名称
        :type model_name: str
        :return: 配置好的logger实例
        :rtype: logger
        """
        # 清理模型名称作为文件名
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # 如果已经存在该模型的logger，直接返回
        if safe_model_name in self._model_loggers:
            return logger.bind(model=model_name)
        
        # 创建模型特定的日志目录
        model_dir = self.model_log_dir / safe_model_name
        model_dir.mkdir(exist_ok=True)
        
        # 详细格式用于模型日志
        model_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>MODEL:{extra[model]}</magenta> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 添加模型特定的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_log_file = model_dir / f"{safe_model_name}_{timestamp}.log"
        
        handler_id = logger.add(
            model_log_file,
            format=model_format,
            level="DEBUG",
            rotation="50 MB",  # 模型日志单文件最大50MB
            retention="15 days",  # 保留15天
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            filter=lambda record: record["extra"].get("model") == model_name
        )
        
        # 添加模型实时日志（不轮转，用于调试）
        current_log_file = model_dir / "current.log"
        logger.add(
            current_log_file,
            format=model_format,
            level="DEBUG",
            backtrace=True,
            diagnose=True,
            enqueue=True,
            filter=lambda record: record["extra"].get("model") == model_name
        )
        
        self._model_loggers[safe_model_name] = handler_id
        
        model_logger = logger.bind(model=model_name)
        model_logger.info(f"为模型 {model_name} 创建专用日志")
        model_logger.info(f"模型日志文件: {model_log_file}")
        
        return model_logger
    
    def get_main_logger(self) -> 'logger':
        """
        获取主logger实例
        
        :return: 主logger实例
        :rtype: logger
        """
        return logger
    
    def cleanup_model_logger(self, model_name: str) -> None:
        """
        清理特定模型的logger（当模型停止时调用）
        
        :param model_name: 模型名称
        :type model_name: str
        :return: None
        :rtype: None
        """
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        if safe_model_name in self._model_loggers:
            handler_id = self._model_loggers[safe_model_name]
            logger.remove(handler_id)
            del self._model_loggers[safe_model_name]
            logger.info(f"已清理模型 {model_name} 的日志处理器")
    
    def set_debug_mode(self, is_debug: bool = True) -> None:
        """
        快速切换到调试模式
        
        :param is_debug: 是否启用调试模式
        :type is_debug: bool
        :return: None
        :rtype: None
        """
        level = "DEBUG" if is_debug else "INFO"
        self.set_console_level(level)
        logger.info(f"日志模式已切换为: {'DEBUG' if is_debug else 'NORMAL'}")


# 全局单例实例
_log_manager = LoguruManager()

# 便捷函数
def get_main_logger():
    """
    获取主logger
    
    :return: 主logger实例
    :rtype: logger
    """
    return _log_manager.get_main_logger()

def get_model_logger(model_name: str):
    """
    获取模型特定的logger
    
    :param model_name: 模型名称
    :type model_name: str
    :return: 模型logger实例
    :rtype: logger
    """
    return _log_manager.get_model_logger(model_name)

def set_console_level(level: str):
    """
    设置控制台日志级别
    
    :param level: 日志级别
    :type level: str
    :return: None
    :rtype: None
    """
    _log_manager.set_console_level(level)

def set_debug_mode(is_debug: bool = True):
    """
    设置调试模式
    
    :param is_debug: 是否启用调试模式
    :type is_debug: bool
    :return: None
    :rtype: None
    """
    _log_manager.set_debug_mode(is_debug)

def cleanup_model_logger(model_name: str):
    """
    清理模型logger
    
    :param model_name: 模型名称
    :type model_name: str
    :return: None
    :rtype: None
    """
    _log_manager.cleanup_model_logger(model_name)

# 导出主logger实例供直接使用
logger = get_main_logger() 
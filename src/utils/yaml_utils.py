from typing import Dict, Any, Optional, LiteralString
import yaml

class YAMLConfigManager:
    @staticmethod
    def read_yaml(file_path: str | LiteralString) -> Optional[Dict[str, Any]]:
        """
        读取 YAML 配置文件并返回数据。

        Args:
            file_path (str): YAML 文件的路径

        Returns:
            Optional[Dict[str, Any]]: 配置数据（字典类型），如果读取失败则返回 None

        Raises:
            yaml.YAMLError: 当 YAML 解析出错时
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
            return None

    @staticmethod
    def write_yaml(file_path: str | LiteralString, data: Dict[str, Any]) -> None:
        """
        将数据写入 YAML 配置文件。

        Args:
            file_path (str): YAML 文件的路径
            data (Dict[str, Any]): 要写入的数据（字典类型）

        Raises:
            yaml.YAMLError: 当 YAML 写入出错时
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                yaml.dump(data, file, default_flow_style=False, allow_unicode=True, indent=4)
        except yaml.YAMLError as e:
            print(f"Error writing to YAML file: {e}")

    @classmethod
    def update_yaml(cls, file_path: str | LiteralString, key: str, value: Any) -> None:
        """
        更新 YAML 配置文件中的特定项。

        Args:
            file_path (str): YAML 文件的路径
            key (str): 键（字典中的路径，如 'server.port'）
            value (Any): 新的值

        Raises:
            KeyError: 当指定的键路径不存在时
        """
        config = cls.read_yaml(file_path)
        if config is not None:
            keys = key.split('.')
            temp = config
            for k in keys[:-1]:
                if k not in temp:
                    raise KeyError(f"Key path '{key}' does not exist in the config.")
                temp = temp[k]
            temp[keys[-1]] = value
            cls.write_yaml(file_path, config)

    @classmethod
    def delete_yaml_item(cls, file_path: str | LiteralString, key: str) -> None:
        """
        删除 YAML 配置文件中的某项。

        Args:
            file_path (str): YAML 文件的路径
            key (str): 键（字典中的路径，如 'database'）

        Raises:
            KeyError: 当指定的键路径不存在时
        """
        config = cls.read_yaml(file_path)
        if config is not None:
            keys = key.split('.')
            temp = config
            for k in keys[:-1]:
                if k not in temp:
                    raise KeyError(f"Key path '{key}' does not exist in the config.")
                temp = temp[k]
            if keys[-1] in temp:
                del temp[keys[-1]]
                cls.write_yaml(file_path, config)
            else:
                raise KeyError(f"Key '{keys[-1]}' does not exist in the config.")

    @staticmethod
    def load_yaml_string(yaml_string:str) -> Optional[Dict[str, Any]]:
        """
        从 YAML 字符串加载数据。

        Args:
            yaml_string (str): YAML 格式的字符串

        Returns:
            Optional[Dict[str, Any]]: 配置数据（字典类型），如果加载失败则返回 None

        Raises:
            yaml.YAMLError: 当 YAML 解析出错时
        """
        try:
            config = yaml.safe_load(yaml_string)
            return config
        except yaml.YAMLError as e:
            print(f"Error loading YAML string: {e}")
            return None
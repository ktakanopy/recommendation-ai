from abc import ABC, abstractmethod


class LoggerPort(ABC):
    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def exception(self, msg: str, *args, **kwargs) -> None:
        pass

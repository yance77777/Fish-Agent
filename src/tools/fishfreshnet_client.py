"""
FishFreshNetV1 模型客户端
用于调用模型API进行新鲜度分类
"""
import os
import requests
from typing import Dict, Any, Optional
import logging
import threading
import time

logger = logging.getLogger(__name__)


class FishFreshNetClient:
    """FishFreshNetV1 API客户端"""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        初始化客户端
        
        Args:
            base_url: API服务地址。未配置时客户端会返回降级结果。
        """
        self.base_url = (base_url or os.getenv("FISHFRESHNET_API_URL") or "").rstrip("/")
        self.timeout = 30
        self.max_retries = 3

    def _is_configured(self) -> bool:
        return bool(self.base_url)

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        if not self._is_configured():
            raise RuntimeError("FISHFRESHNET_API_URL is not configured")

        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(0.5 * (2 ** attempt))
        raise last_error or RuntimeError("FishFreshNet API request failed")
    
    def predict_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        从URL预测新鲜度
        
        Args:
            image_url: 图片URL
        
        Returns:
            预测结果字典
        """
        if not image_url.startswith(("http://", "https://")):
            return self.predict_from_file(image_url)
        try:
            response = self._request("POST", "/predict_url", json={"image_url": image_url})
            return response.json()
        except RuntimeError as e:
            logger.error(str(e))
            return self._fallback_result(str(e))
        
        except requests.exceptions.Timeout:
            logger.error("API请求超时")
            return self._fallback_result("API请求超时")
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return self._fallback_result(f"API请求失败: {str(e)}")
    
    def predict_from_file(self, image_path: str) -> Dict[str, Any]:
        """
        从本地文件预测新鲜度
        
        Args:
            image_path: 本地图片路径
        
        Returns:
            预测结果字典
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self._request("POST", "/predict", files=files)
                return response.json()
        except RuntimeError as e:
            logger.error(str(e))
            return self._fallback_result(str(e))
        
        except requests.exceptions.Timeout:
            logger.error("API请求超时")
            return self._fallback_result("API请求超时")
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return self._fallback_result(f"API请求失败: {str(e)}")
    
    def predict_with_gradcam_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        从URL预测新鲜度并生成Grad-CAM
        
        Args:
            image_url: 图片URL
        
        Returns:
            包含预测结果和热力图base64的字典
        """
        if not image_url.startswith(("http://", "https://")):
            return self.predict_with_gradcam_from_file(image_url)
        try:
            response = self._request("POST", "/gradcam_url", json={"image_url": image_url})
            return response.json()
        except RuntimeError as e:
            logger.error(str(e))
            return self._fallback_result(str(e))
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return self._fallback_result(f"API请求失败: {str(e)}")

    def predict_with_gradcam_from_file(self, image_path: str) -> Dict[str, Any]:
        try:
            with open(image_path, "rb") as f:
                response = self._request("POST", "/predict_with_gradcam", files={"file": f})
                return response.json()
        except RuntimeError as e:
            logger.error(str(e))
            return self._fallback_result(str(e))
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: {e}")
            return self._fallback_result(f"API请求失败: {str(e)}")
    
    def health_check(self) -> bool:
        """
        检查API服务是否可用
        
        Returns:
            服务是否可用
        """
        if not self._is_configured():
            logger.info("FishFreshNet API URL is not configured")
            return False
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def _fallback_result(self, error_message: str) -> Dict[str, Any]:
        """
        返回兜底结果
        
        Args:
            error_message: 错误信息
        
        Returns:
            兜底结果字典
        """
        return {
            "freshness_level": "未知",
            "freshness_label": -1,
            "confidence_score": 0.0,
            "all_probabilities": {
                "高度新鲜": 0.0,
                "新鲜": 0.0,
                "不新鲜": 0.0
            },
            "description": f"模型预测失败: {error_message}",
            "error": error_message
        }


# 全局客户端实例
_client = None
_client_lock = threading.Lock()


def get_client() -> FishFreshNetClient:
    """获取全局客户端实例"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = FishFreshNetClient()
    return _client


if __name__ == "__main__":
    # 测试客户端
    client = get_client()
    
    # 健康检查
    if client.health_check():
        print("✅ API服务可用")
    else:
        print("❌ API服务不可用")

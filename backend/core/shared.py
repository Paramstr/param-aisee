"""Shared resources and dependency injection container"""
import asyncio
import concurrent.futures
from typing import Optional


class SharedResources:
    """Container for shared system resources"""
    
    def __init__(self):
        # Shared thread pools
        self.cpu_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.io_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        
        # Event loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def initialize(self):
        """Initialize shared resources"""
        self.loop = asyncio.get_running_loop()
        
        # Create specialized thread pools
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix="cpu-"
        )
        self.io_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="io-"
        )
    
    async def cleanup(self):
        """Clean up resources"""
        if self.cpu_pool:
            self.cpu_pool.shutdown(wait=True)
            self.cpu_pool = None
        
        if self.io_pool:
            self.io_pool.shutdown(wait=True)
            self.io_pool = None


class DependencyContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self.shared = SharedResources()
        self._audio_processor = None
        self._vision_processor = None
        self._llm_processor = None
        self._task_manager = None
    
    async def initialize(self):
        """Initialize all dependencies"""
        await self.shared.initialize()
        
        # Import here to avoid circular imports
        from .audio import AudioProcessor
        from .vision import VisionProcessor
        from .llm import LLMProcessor
        from .tasks import TaskManager
        
        # Create instances with dependencies
        self._audio_processor = AudioProcessor(
            cpu_pool=self.shared.cpu_pool,
            loop=self.shared.loop
        )
        
        self._vision_processor = VisionProcessor()
        
        self._llm_processor = LLMProcessor(
            io_pool=self.shared.io_pool
        )
        
        self._task_manager = TaskManager(
            audio_processor=self._audio_processor,
            vision_processor=self._vision_processor,
            llm_processor=self._llm_processor
        )
        
        # Initialize processors
        await self._audio_processor.initialize()
    
    async def cleanup(self):
        """Clean up all dependencies"""
        if self._task_manager:
            await self._task_manager.stop_all_tasks()
        
        await self.shared.cleanup()
    
    @property
    def audio_processor(self):
        return self._audio_processor
    
    @property
    def vision_processor(self):
        return self._vision_processor
    
    @property
    def llm_processor(self):
        return self._llm_processor
    
    @property
    def task_manager(self):
        return self._task_manager


# Global container instance
container = DependencyContainer() 
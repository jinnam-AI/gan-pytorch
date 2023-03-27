import inspect

class Registry:
    def __init__(self,name:str) -> None:
        self._name = name
        self._module_dict = {}
        
    def get(self,key:str):
        return self._module_dict.get(key)
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def _register_module(self,cls,cls_name:str=None):
        
        if not inspect.isclass(cls):
            raise TypeError('module must be a class,'f'but got {type(cls)}')
        
        if cls_name is None:
            cls_name == cls.__name__
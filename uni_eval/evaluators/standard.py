import logging
from typing import List, Dict, Any
from tqdm import tqdm

from uni_eval.registry import EVALUATORS
from uni_eval.evaluators.base import BaseEvaluator
from uni_eval.models.base import BaseModel
from uni_eval.datasets.base import BaseDataset

logger = logging.getLogger(__name__)

@EVALUATORS.register_module()
class StandardEvaluator(BaseEvaluator):

    def evaluate(self, model: BaseModel, dataset: BaseDataset, **kwargs) -> List[Dict[str, Any]]:
        logger.info(f"Starting evaluation on {len(dataset)} samples...")
        
        prompts = []
        for item in dataset:
            if 'prompt' not in item:
                raise KeyError(f"Dataset item missing 'prompt' field: {item}")
            prompts.append(item['prompt'])

        batch_size = kwargs.get('batch_size', 100)
        outputs = []
        
        total = len(prompts)
        with tqdm(total=total, desc="Generating") as pbar:
            for i in range(0, total, batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_outputs = model.generate(batch_prompts, **kwargs)
                outputs.extend(batch_outputs)
                pbar.update(len(batch_prompts))

        results = []
        for item, output in zip(dataset, outputs):
            result_item = item.copy() if isinstance(item, dict) else dict(item)
            result_item['prediction'] = output
            results.append(result_item)
            
        return results

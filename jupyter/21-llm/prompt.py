LLM_PROMPT_PREFIX = """Suppose you have a data science notebook and want to extract pipeline components based on their semantic purposes. There are two requirements. First, each component should contain consecutive code, one more more lines, in the notebook. You should output one code cell for each component. You cannot modify, swap, or exclude any code. Second, each component should represent a specific stage in the data science process. Components can have the same stage. The example stages are:
- data acquisition (such as load, collect, obtain, capture, survey)
- data preparation (such as explore, wrangle, clean, filter, organize)
- storage (such as preserve, archive, warehouse, log, recycle)
- feature engineering (such as feature, label, annotate)
- modeling (such as classify, cluster, mine, analyze, process)
- training (such as tune, optimize)
- evaluation (such as validate, test, verify, review)
- prediction (such as discover, derive, determine)
- interpretation (such as transform, visualize, render, translate, explain)
- communication (such as transfer, share, distribute, transmit, publish)

"""

def make_llm_prompt(encoding: dict):
    result = LLM_PROMPT_PREFIX
    
    if len(encoding['func_defs']) == 0:
        result += 'The notebook has no global function definitions.\n'
    else:
        result += 'The notebook has the following global function definitions:\n'
        for fdef in encoding['func_defs']:
            result += '### FUNCTION DEFINITION\n'
            result += fdef
            result += '\n'
            result += '### FUNCTION DEFINITION END HERE\n'

    result += 'And the notebook code is:\n'
    result += '### NOTEBOOK CODE\n'
    result += "\n".join(encoding['code_lines'])
    result += '\n'
    result += '### NOTEBOOK CODE END HERE\n'

    result += 'Hence, the pipeline components are:\n'
    result += '### COMPONENT'

    return result

def make_llm_prompt_cot(encoding: dict):
    return make_llm_prompt(encoding) + "\nLet's think step by step."

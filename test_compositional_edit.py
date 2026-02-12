import os
import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset, CompositionalCaptionDataset, CompositionalDataset_RAG_70, CompositionalDataset_RAG_50, CompositionalDataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams, OURSMultimodalHparams, VisEditHparams, WISEMultimodalHyperParams, LORAMultimodalHparams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer
import sys
from datetime import datetime

'''
Baselines
'''

####################### FT ##########################

def test_LLaVA_FT_comp():
    '''
    FT baseline for LLaVA compositional editing
    Uses CompositionalDataset (no Query Decomposition) + CCKEB_eval.json
    '''
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava_compositional_edit.yaml')
    eval_ds = CompositionalDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=500)

def test_MiniGPT4_FT_comp():
    '''
    FT baseline for MiniGPT4 compositional editing
    Uses CompositionalDataset (no Query Decomposition) + CCKEB_eval.json
    '''
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4_compositional_edit.yaml')
    eval_ds = CompositionalDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num,test_num=500)

####################### LoRA ##########################

def test_LLaVA_one_lora_comp():
    '''
    LoRA baseline for LLaVA compositional editing (single LoRA for both visual and textual)
    Uses CompositionalDataset (no Query Decomposition) + CCKEB_eval.json
    '''
    hparams = LORAMultimodalHparams.from_hparams('hparams/LORA/llava_compositional_one_lora.yaml')
    eval_ds = CompositionalDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=500)

def test_MiniGPT4_one_lora_comp():
    '''
    LoRA baseline for MiniGPT4 compositional editing (single LoRA for both visual and textual)
    Uses CompositionalDataset (no Query Decomposition) + CCKEB_eval.json
    '''
    hparams = LORAMultimodalHparams.from_hparams('hparams/LORA/minigpt4_compositional_one_lora.yaml')
    eval_ds = CompositionalDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_compositional_ft(log=True, gap_num=gap_num, test_num=500)

####################### SERAC ##########################

def test_LLaVA_SERAC_comp():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/llava.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[4,5])

def test_MiniGPT4_SERAC_comp():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[6,7])

####################### WISE ##########################

def test_LLaVA_WISE_comp():
    hparams = WISEMultimodalHyperParams.from_hparams('hparams/WISE/llava.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[0,1])

def test_MiniGPT4_WISE_comp():
    hparams = WISEMultimodalHyperParams.from_hparams('hparams/WISE/minigpt4.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[2,3])

'''
OURS
'''

####################### Stage 2: Knowledge Connector Training  ##########################

def train_LLaVA_OURS_stage2():
    """Train Knowledge Connector for LLaVA with 70% RAG accuracy threshold (Stage 2)"""
    if gap_num == 0:
        hparams = LORAMultimodalHparams.from_hparams('hparams/OURS/stage2/llava_train_compositional_connector_rag_70.yaml')
        train_ds = CompositionalDataset_RAG_70(train_comp_final_json_path, config=hparams, hop=hop)
        trainer = MultimodalTrainer(
            config=hparams,
            train_set=train_ds,
            val_set=train_ds
        )
        trainer.test_sequencial_compositional_connector_attention_rag_70(log=True, gap_num=gap_num, test_num=500)
    else: 
        exit()

def train_MiniGPT4_OURS_stage2():
    """Train Knowledge Connector for MiniGPT4 with 50% RAG accuracy threshold (Stage 2)"""
    if gap_num == 0:
        hparams = LORAMultimodalHparams.from_hparams('hparams/OURS/stage2/minigpt4_train_compositional_connector_rag_50.yaml')
        train_ds = CompositionalDataset_RAG_50(train_comp_final_json_path, config=hparams, hop=hop)
        trainer = MultimodalTrainer(
            config=hparams,
            train_set=train_ds,
            val_set=train_ds
        )
        trainer.test_sequencial_compositional_connector_attention_rag_50(log=True, gap_num=gap_num, test_num=500)
    else: 
        exit()

####################### Test ##########################

def test_LLaVA_OURS_comp():
    hparams = OURSMultimodalHparams.from_hparams('hparams/OURS/llava.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[0,1])

def test_MiniGPT4_OURS_comp():
    hparams = OURSMultimodalHparams.from_hparams('hparams/OURS/minigpt4.yaml')
    eval_ds = CompositionalCaptionDataset(eval_comp_final_json_path, config=hparams, hop=hop)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    trainer.test_sequencial_multi_gpus(log=True, gap_num=gap_num, test_num=500, comp=True, training=False, gpus=[2,3])


if __name__ == "__main__":
    function_name = sys.argv[1]

    #train_comp_json_path = 'datasets/train_comp.json'
    #eval_comp_new_json_path = 'datasets/eval_comp_0411.json' # 내가 검토후 데이터셋(Jiyun)

    train_comp_final_json_path = 'datasets/CCKEB_train.json'
    eval_comp_final_json_path = 'datasets/CCKEB_eval.json' 

    if function_name not in globals() or not callable(globals()[function_name]):
        print(f"Error: Function '{function_name}' does not exist.")
        sys.exit(1)

    for gap_num in [0, 10, 20, 50, 100]:
        globals()[function_name]()

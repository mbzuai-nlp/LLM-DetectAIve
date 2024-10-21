# LLM-DetectAIve

*A Tool for Fine-Grained Machine-Generated Text Detection*

<a href="https://arxiv.org/abs/2405.05583"><img src="https://img.shields.io/badge/arXiv-2405.05583-B31B1B" alt="arXiv"></a>

<a href="https://huggingface.co/spaces/raj-tomar001/LLM-DetectAIve">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" 
       alt="Hugging Face Logo" style="width: 50px; object-fit: contain;">
</a>

[Try it yourself!](tps://huggingface.co/spaces/raj-tomar001/LLM-DetectAIve)

[Video demo](https://www.youtube.com/watch?v=E8eT_bE7k8c&feature=youtu.be)


---

## Overview  

LLM-DetectAIve is a system designed for fine-grained MGT detection. It is aims to classify texts from a variety of topics into four categories: human-written, machine-generated, machine-written machine-humanized, and human-written machine-polished.

LLM-DetectAIve can effectively identify the authorship of textual content, proving its usefulness in enhancing integrity in education, academia, and other domains. LLM-DetectAIve is publicly accessible on hugging face

---

## Demo Interface

The demonstration for LLM-DetectAIve has two tabs

### 1. Automatic Text Detection
![Demo Interface 2](assets/Interface1.png)

    
Lets the user input any text (> 50 words) to get a prediction from LLM-DetectAIve.


### 2. Human Detector Playground

![Demo Interface 2](assets/Interface2.png)

Lets the user see if they can correctly label data given a random sample text.

## Data Generated

Across LLMs over the four classes:  
I. Human-Written  
II. Machine-Generated  
III. Machine-Written Machine-Humanized  
IV. Human-Written Machine-Polished  
 

![Data Generated Table](assets/data_generated_table.png)

<!-- <img src="assets\data_generated_table.png" width="100%"> -->
For row II + III + IV, the data is approximately uniformly distributed across the three classes.


---


## Cite
If you use LLM-DetectAIve in your research cite as:
```bibtex
@article{abassy2024llmdetectaivetoolfinegrainedmachinegenerated,
      title = {LLM-DetectAIve: a Tool for Fine-Grained Machine-Generated Text Detection}, 
      author = {Mervat Abassy and Kareem Elozeiri and Alexander Aziz and Minh Ngoc Ta and Raj Vardhan Tomar and Bimarsha Adhikari and Saad El Dine Ahmed and Yuxia Wang and Osama Mohammed Afzal and Zhuohan Xie and Jonibek Mansurov and Ekaterina Artemova and Vladislav Mikhailov and Rui Xing and Jiahui Geng and Hasan Iqbal and Zain Muhammad Mujahid and Tarek Mahmoud and Akim Tsvigun and Alham Fikri Aji and Artem Shelmanov and Nizar Habash and Iryna Gurevych and Preslav Nakov},
      journal = {arXiv preprint arXiv:2408.04284},
      year = {2024}

}

```

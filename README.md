# DHBW_BachelorThesis_Code

The Code repository for my own Bachelorthesis in 2021. 
## Abstract 

 As AI models have become more and more complex in recent years, the need for explanations has become greater to increase comprehensibility and trust in the models as trust is becoming increasingly important since AI models are starting to influence decision-making in most aspects of our lives. At the same time, the concept of attention in AI models emerged in the field of NLP, which makes it possible for models to focus on the important things in a complex input and helps us to understand on which things the algorithm has focused. After the good results in NLP, attempts were also made to transfer the attention concepts to the field of CV. In this bachelor thesis, the newly developed attention-based vision transformer is applied, and its results are visualised and explained with the help of calculated attention. This novel explainer is compared with an already established explainer called LIME. The comparison and the quality determination are carried out by using newly published quantitative metrics. This bachelor thesis is the first paper to-date to independently evaluate these new quantitative evaluation metrics of expainability. A closer look at the metrics revealed that they have a logical basis and, therefore, can measure the quality of explainers. However, the attention-based explainer proposed in this bachelor thesis performed slightly worse than the LIME. This could be partly caused by the functionality of the metrics, the slightly worse general performance of the model, and the explainer itself. With a little fine-tuning, however, most of the problems found can be solved.

In summary, the intrinsic attention-based explainer developed in this bachelor thesis is a good explainer with much potential. It combines attention with the explanation of AI models and thus contributes to more comprehensibility and trust in those models. 

## Execute Code

Create Virtual Environment
```python -m venv venv```

On Windows, run:
```venv\Scripts\activate.bat```

On Unix or MacOS, run:
```source venv/bin/activate```


Install requiered packages:
```python -m pip install -r requirements.txt```

Run Code:
```python main.py```


## Acknowladegements:

- Dataset ImageNette:
    - https://github.com/fastai/imagenette
- ViT:
    - https://github.com/lucidrains/vit-pytorch



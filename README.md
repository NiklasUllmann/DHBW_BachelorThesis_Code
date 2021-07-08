# DHBW_BachelorThesis_Code

The Code repository for my own Bachelorthesis in 2021. 

Topic: "Using attention techniques for explainability of deep learning models in computer vision"

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

## Other Stuff:

imagenette_map = { 
    "n01440764" : "tench", 0
    "n02102040" : "springer",1
    "n02979186" : "casette_player",2
    "n03000684" : "chain_saw",3
    "n03028079" : "church",4
    "n03394916" : "French_horn",5
    "n03417042" : "garbage_truck",6
    "n03425413" : "gas_pump",7
    "n03445777" : "golf_ball",8
    "n03888257" : "parachute"9
}

torch.Size([256, 3, 320, 320])
torch.Size([256, 6, 316, 316])
torch.Size([256, 6, 158, 158])
torch.Size([256, 6, 154, 154])
torch.Size([256, 6, 77, 77])

## Acknowladegements:

- Dataset ImageNette:
    - https://github.com/fastai/imagenette
- ViT:
    - https://github.com/lucidrains/vit-pytorch
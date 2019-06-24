# Visual-Question-Answering

Ce repository fournit une implémentation du papier : [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)

Le code de ce repository peut être directement utilisé sur [ce colab](https://colab.research.google.com/drive/1nIXGzbOvHQPIbDKqcmBwccH21K7GPtRE). Le fichier ipynb utilisé peut également être téléchargé [ici](https://github.com/numediart/Visual-Question-Answering/blob/master/VQA_handsonai.ipynb).

Le Visual Question Answering (VQA) est une discipline informatique ou la machine doit répondre à une question formée en langage naturel (et non pas en mots clé) a propos d'une scène visuelle. L'entrée du modèle est donc une question et une image, la sortie est une classe qui correspond a un mot ou un groupe de mot (i.e. il existe la classe "pomme" et la classe "feu rouge").

<p align="center">
<img src="https://visualqa.org/static/img/challenge.png" width="500px" align="center" />
</p>

### Requirements

Python3.6<br/>
torch 1.1.0<br/>
torchvision 0.2.2<br/>
Pillow 6.0.0<br/>
tqdm<br/>

Dézipper data.zip dans le dossier data


|         | GPU     | CPU  | RAM |
|:-------------:|:-------------:|:-------------:|:-------------:|
| Training     | 10h | - | 50 Go |
| Prediction     | 6s | - | 1Mo |



### Training
Premièrement, télécharger les images MSCOCO [train2014](http://images.cocodataset.org/zips/train2014.zip) et [val2014](http://images.cocodataset.org/zips/val2014.zip) et placer les dossiers dézippés dans "data".

Ensuite, on peut lancer un nouvel entrainement avec la commande suivante :
```
python main.py  
```
L'entrainement s'arrête automatiquement après 5 epochs sans amélioration.
Le meilleur checkpoint sera stocké dans le dossier "ckpt".

### Evaluation
L'évaluation est possibile avec la commande suivante. Le fichier pré-entrainé est fourni dans ce repository.
```
python main.py --eval True --ckpt ckpt/model_0.5590.pth
```
La sortie devrait être:
```
loading dictionary from data/dictionary.pkl
Loading data/val_resnet_layer3.pkl
	eval score: 55.90 (92.66)
```
Ce qui signifie une précision de bonne réponse à 56%. Plus en détail, voici les sous-scores:

| Yes/No        | Number     | Other  | Overall |
|:-------------:|:-------------:|:-------------:|:-------------:|
| 76.6%      | 36.2% | 49.5% | 55.90% |

Il faut noter que les mauvaises réponses ne sont pas forcément médiocres ou absurdes. Répondre fille au lieu de femme n'est pas correct mais n'est pas trop mauvais, selon l'utilisation qu'on veut faire du modèle.

### Prediction

Il est possible d'utiliser le modèle sur ses propres données. Voici la commande a utiliser avec le fichier example.jpg fourni dans ce repository:
```
python main.py  --inference True \
		--ckpt ckpt/model_0.5590.pth \
		--image example.jpg \
		--question "What is the man holding ?"
```
<img src="https://github.com/numediart/Visual-Question-Answering/blob/master/example.jpg?raw=true&s=100" width="200" />

La sortie est :
```
loading dictionary from data/dictionary.pkl
Question: What is the man holding ?
Answer: banana
```

D'autres exemples :

```
loading dictionary from data/dictionary.pkl
Question: What is the color of the man's hair ?
Answer: brown
```

```
loading dictionary from data/dictionary.pkl
Question: Is the man handsome and smart ?
Answer: yes
```

# Završni rad

Završni rad preddiplomskog studija Računarstvo na Fakultatu elektrotehnike i računarstva sveučilišta u Zagrebu

## Raspored direktorija

Sav kod vezan usko za završni rad raspoređen je u direktorije notebooks i scripts.
Ostali kod kao npr. vlastitia implementacija CelebA razreda za učitavanje podatkovnog skupa nalazi se u direktoriju data,
a vlastita implementacija modela kao i težinske vrijednosti nalaze se u direktoriju models

Odgovarajuće datoteke za rad potrebnih modela nalaze se u direktorijima cascade i mtcnn

## Replikacija

Za relikaciju rezultata potrebno je instalirati PyTorch i OpenCV za python. Ovo se najlakše postiže
koristeći anacondu.

Također je potrebno preuzeti skup [CelebA](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg).
Potrebno je preuzeti samo neporavnate slike, iz direktorija Anno samo list_bbox_celeba.txt i direktorij Eval. Iako ostatak direktorija nije potrebno
preuzeti stuktura direktorija mora biti ista kao i na drive-u.

Za pokretanje MTCNN i CascadeClassifier-a potrebno je kopirati iz direkotrija scripts u korijenski direktorij odgovarajué skripte i pokrenuti ih s odgovarajućim putanjama i opcijama.

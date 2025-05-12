

## Notes pour pyqtgraph
https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html
probablement mieux d'utiliser ImageItem pour montrer l'animation ça sera + propre. DONE

en utilisant un PlotItem : https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotitem.html DONE

et il faut probablement créer une classe pour le layout de la fenêtre.

pour finir de générer le film :
https://github.com/kkroening/ffmpeg-python
with something like : https://stackoverflow.com/a/63640796

## colormap choice
https://www.nature.com/articles/s41467-020-19160-7 

## conda setup : 
```shell
conda install -c conda-forge cupy python-xxhash
conda install --file requirements.txt   
```

## Gpu support
For apple metal this could work : https://github.com/ml-explore/mlx

# L2L2020_tech
Final technical assignment for the Learn2launch Berkeley Program

Build and run with:
```shell
docker-compose up --build
```

After deploying, start to scrape images (~10 mins) and build a training dataset:
```shell script
curl  http://localhost:5002/scrape
```

Then train the model (~10 mins):
```shell script
curl  http://localhost:5002/train
```

Play with your own images:
```shell script
curl -F 'file=@path/picture.jpg' http://localhost:5002/predict
```

Finally, download your model:
```shell script
curl http://localhost:5002/export --output some_file.h5
```


You can interact with the code through the file ```parameters.json```
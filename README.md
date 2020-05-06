# L2L2020_tech
Final technical assignment for the Learn2launch Berkeley Program

Build and run with:
```shell
docker-compose up --build
```

After deploying, start to scrape images (~10 mins) and build a training dataset:
```shell script
curl curl  http://localhost:5002/scrape
```

Then train the model (~10 mins):
```shell script
curl curl  http://localhost:5002/train
```

Finally, play with your own images:
```shell script
curl -F 'file=@path/picture.jpg' http://localhost:5002/predict
```


You can interact with the code through the file ```parameters.json```
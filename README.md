# Bachelor Project P6
## Atomatic Email Detection System

This is a collaborative bachelors project between Anders E. Anders R. & Philip S. D.

### How to run:
##### 1. Train the model
This program requires you to train the model:
```sh
python .\src\train.py
```

##### 2. Change the paths
Now that you have a model, make sure to change the paths to point to the actual folder.
E.g:

```py
# in /server/api_server.py
model_path = os.path.join("models", "bert_lr2e-05_ep1_0403-1444") # Change to your model name.
```

##### 3. Setup docker and run ngrok

To setup docker, make sure you have docker desktop opened and installed, then run:

```sh
docker build -t phishing_detection .
```

and:

```sh
docker run -p 5000:5000 phishing_detection
```

To connect the add-on to your container we use `ngrok`, which can be installed from: https://ngrok.com/.

When it's installed, run:

```sh
ngrok http 5000
```

(Replace `5000` if your container exposes a different port).

##### 4. Set up the Apps Script

1. Get the Public URL:
   `ngrok` will display a public HTTPS URL (e.g., `https://abcdef123456.ngrok.io`). This URL tunnels traffic to your `localhost:5000`.

2. Update `main.gs`:
   Copy the `ngrok` HTTPS URL and paste it into the ``LOCAL_API_ENDPOINT`` constant in your `main.gs` file. Remember to add `/predict` at the end.

   ```js
   const LOCAL_API_ENDPOINT = "https://abcdef123456.ngrok.io/predict"; // USE YOUR ACTUAL NGROK URL
   ```
3. Go to https://script.google.com/home and create a new project, or edit existing project.
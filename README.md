# Project-2-Eagle-Eye
A tool to predict the popularity of a new song based on its various metrics (danceability, energy, tempo, etc.) before its been released. This has the ability to integrate with spotify api to gather various popularity metrics with the latest up to date information, for a list of songs to train on. 

# Requirements:
* ffmpeg (libary used for scraping audio from youtube)
* yt-dlp (tool used for scraping the audio from youtube)
* python 3 

# Installation Instructions
* Run the following command from the app root directory to install the required python packages.

  ``` pip install -r requirements.txt ```

# Run the application
To run the application run the following command from the root direcotry of the application

``` FLASK_APP=src/app flask run --port=5050 --host=localhost ```

You can then access the application by visiting http://localhost:5050 within your web browser. 

The reason we specify port 5050 is because mac has started running airplay on port 5000 which is what the default port is that flask uses when starting an application. If apple airplay is already using the port. You will get a "port already in use" error.

Also keep in mind, the callback URL has to be put in the spotify for it to properly work with the spotify api. In this case with the port being assigned on 5050 and running on localhost. You would put http://localhost:5050/callback. In the callback section when creating the spotify API token. 

If you want to use a custom domain. You will need to add the custom domain into the callback URL's so it would be http://www.customdomain.com:5050/callback as an example. 

You would then start up the app using the custom domain name instead of localhost. 

``` FLASK_APP=src/app flask run --port=5050 --host=www.customdomain.com ```

If you want it to be running on https (secure) you will need to implement certificates and run it on port 443 instead of 5050 or have it behind a load balancer. This is an entirely different ballpark of complexity. You will need to do your own research to get that working, its too much to go into on this simple README. 
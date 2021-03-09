## It matters

### Intro
The IT industry is improving at a rapid pace. Every year new, revolutionary and powerful devices are introduced to the public. However, the way we interact with these devices has not changed in ages. We spend countless hours looking at our computer screens, seriously hurting our eyes. We don't take breaks. Most of us work from cafes where we sit in an unnatural position for an extremely long time. We claim to take care of our health but neglect the impact of the activity that we do the most. 

### About the project
My vision is getting worse and worse, my eyes often feel tired after a long working day and on top of all that my back hurts. I can't imagine my life without a computer yet every additional hour of usage harms my body. Hence, I decided to create a piece of software that would monitor my daily computer interaction. By using it, I would have a clear view on my progress of getting rid of my unhealthy habits. The app's purpose is to be a quiet informer, so I won't receive any notifications and won't be disturbed while working.

### What it does
* monitors user's face and checks if the user is taking breaks by looking away from the PC screen. It uses [20-20-20 rule](https://opto.ca/health-library/the-20-20-20-rule) template.
* Checks if the user is too close to the screen
* monitors user's body position and checks if his posture is correct
* User can access all statistics about his screen time, face and body position through a simple web client.

### Implementation
* Python client (Monitoring, sending data to the server)
* NodeJS server (storage, data interpretation, main logic)
* React app (graphs, user interaction)

### How to run the web app
0. It is not necessary to run the web app on your localhost. I have the server running on Heroku. I can easily set up a user account for you and send you the credentials. Anyway, if you'd like to run your own server, proceed to the next step.
1. Clone the latest code from `it-matters-web` repository.
2. run docker containers by `docker-compose up`
3. Then, run `cd database && yarn migrate:latest` in order to execute all db migrations.
4. [optional] I have created a default user account (also included in the python client). If you would like to use it, then run `yarn seed:run` in the same folder.

### How to run the Python client
1. Clone the latest code.
2. Install all requirements to your own environment or create a virtual one.
3. If you are not using the default user, you need to set the correct parameters in the `main.py` file (USER_ID, API_KEY).
4. run `python3 main.py`
5. It is recommended to run the python script every time your system is booted. I personally use crontab.

### Face detection
As part of the project I have built a copy of Viola Jones algorithm. The code can be found in `custom_face_detection` in `it-matters-client` repository.
My model which uses 30 weak classifiers was trained on 60000 pictures and achieved 81% accuracy on the testing data.


<!-- TODO: https://www.rockyourcode.com/docker-postgres-knex-setup/  docker-compose rm -f     docker-compose up --build  -->

FROM node:20.18.0-alpine

WORKDIR /app

COPY . .

RUN npm install

RUN chmod +x start.sh

CMD ["sh", "start.sh"]
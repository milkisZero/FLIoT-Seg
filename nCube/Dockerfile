FROM node:20.18.0-alpine

WORKDIR /app

COPY . .

RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    py3-pip

RUN npm install -g pm2

RUN npm install

RUN chmod +x start.sh

CMD ["sh", "start.sh"]
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
# Place config in /templates/ so nginx's entrypoint runs envsubst on it at
# startup, substituting ${LARA_API_KEY} from the container environment before
# nginx reads the config.
COPY nginx.conf /etc/nginx/templates/default.conf.template

EXPOSE 5173

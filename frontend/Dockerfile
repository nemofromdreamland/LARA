FROM node:22-alpine AS builder

WORKDIR /app

# Copy lockfile first so npm ci can reproduce the exact dependency tree.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:stable-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
# Place config in /templates/ so nginx's entrypoint runs envsubst on it at
# startup, substituting ${LARA_API_KEY} from the container environment before
# nginx reads the config.
COPY nginx.conf /etc/nginx/templates/default.conf.template
# Startup guard: fail fast if LARA_API_KEY is not set before nginx starts.
COPY docker-entrypoint.sh /usr/local/bin/lara-entrypoint.sh
RUN chmod +x /usr/local/bin/lara-entrypoint.sh

EXPOSE 5173

ENTRYPOINT ["/usr/local/bin/lara-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]

FROM node:22-alpine AS builder

WORKDIR /app

# Copy lockfile first so npm ci can reproduce the exact dependency tree.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ARG VITE_API_KEY
ENV VITE_API_KEY=$VITE_API_KEY
RUN npm run build

FROM nginx:stable-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]

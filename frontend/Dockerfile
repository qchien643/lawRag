FROM node:22-slim

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code
COPY . .

EXPOSE 5173

# Vite dev server — host 0.0.0.0 so Docker can expose it
CMD ["npx", "vite", "--host", "0.0.0.0"]

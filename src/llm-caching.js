import { CacheClient, Configurations, CredentialProvider } from '@gomomento/sdk';
import Redis from 'ioredis';
import { MomentoCache } from 'langchain/cache/momento';
import { RedisCache } from 'langchain/cache/redis';
import { UpstashRedisCache } from 'langchain/dist/cache/upstash_redis.js';
import { OpenAI } from 'langchain/llms/openai';

import './env.js';

async function memory() {
  const llm = new OpenAI({
    cache: true,
  });
}

async function momento() {
  const client = new CacheClient({
    configuration: Configurations.Laptop.v1(),
    credentialProvider: CredentialProvider.fromEnvironmentVariable({
      environmentVariableName: 'MOMENTO_AUTH_TOKEN',
    }),
    defaultTtlSeconds: 60 * 60 * 24,
  });
  const cache = await MomentoCache.fromProps({
    client,
    cacheName: 'langchain',
  });

  const model = new OpenAI({ cache });
}

async function redis() {
  const client = new Redis({});
  const cache = new RedisCache(client);
  const model = new OpenAI({ cache });
}

async function upstashRedis() {
  const cache = new UpstashRedisCache({
    config: {
      url: 'UPSTASH_REDIS_REST_URL',
      token: 'UPSTASH_REDIS_REST_TOKEN',
    },
  });

  const model = new OpenAI({ cache });
}

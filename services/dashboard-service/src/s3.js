import { S3Client } from "@aws-sdk/client-s3";
import "dotenv/config";

const config = (endpoint) => ({
  endpoint,
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
  forcePathStyle: true,
});

export const s3 = new S3Client(config(process.env.S3_ENDPOINT || process.env.S3_PUBLIC_ENDPOINT));
export const publicS3 = new S3Client(
  config(process.env.S3_PUBLIC_ENDPOINT || process.env.S3_ENDPOINT)
);

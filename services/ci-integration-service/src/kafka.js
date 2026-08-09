import { Kafka } from "kafkajs";
import "dotenv/config";

const kafka = new Kafka({
  clientId: "ci-integration-service",
  brokers: [process.env.KAFKA_BROKER],
});

export const producer = kafka.producer();

export async function connectProducer() {
  await producer.connect();
  console.log("[ci-integration-service] Kafka producer connected");
}

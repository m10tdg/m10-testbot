import { Kafka } from "kafkajs";
import "dotenv/config";

const kafka = new Kafka({
  clientId: "document-service",
  brokers: [process.env.KAFKA_BROKER],
});

export const producer = kafka.producer();

// kafkajs requires an explicit connect() before you can send anything.
// We call this once, at service startup, from index.js.
export async function connectProducer() {
  await producer.connect();
  console.log("[document-service] Kafka producer connected");
}

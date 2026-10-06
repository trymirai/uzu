import { createFileRoute } from "@tanstack/react-router";
import { ChatPage } from "@/features/chat/components/chat-page";

export const Route = createFileRoute("/chat/$chatId")({
  component: ChatPage,
});

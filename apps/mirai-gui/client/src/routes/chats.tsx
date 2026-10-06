import { createFileRoute } from "@tanstack/react-router";
import { ChatHistoryPage } from "@/features/chat-history/components/chat-history-page";

export const Route = createFileRoute("/chats")({
  component: ChatHistoryPage,
});

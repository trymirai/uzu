import { useChatStore } from "@/stores/useChatStore";
import { useModelsStore } from "@/stores/useModelsStore";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";
import { getPlatform } from "@/platform/platformSingleton";
import { navigationRequestTypes, type NavigationRequest } from "@/platform/PlatformClient";

type NavigationHandlers = {
  [K in NavigationRequest["type"]]: (req: Extract<NavigationRequest, { type: K }>) => void;
};

export const useAppNavigationEvents = (): void => {
  const createNewChat = useChatStore((s) => s.createNewChat);
  const navigate = useNavigate();

  const openNewChatForModel = useCallback(
    (identifier: string) => {
      const model = useModelsStore.getState().models.find((m) => m.repoId === identifier);
      const modelName = model?.name ?? identifier;
      const newChatId = uuidv4();
      createNewChat(newChatId);
      navigate({
        to: "/chat/$chatId",
        params: { chatId: newChatId },
        search: {
          model: identifier,
          modelName,
          isNew: true,
        },
      });
    },
    [createNewChat, navigate],
  );

  useEffect(() => {
    const handlers: NavigationHandlers = {
      [navigationRequestTypes.newChat]: () => {
        const newChatId = uuidv4();
        createNewChat(newChatId);
        navigate({ to: "/chat/$chatId", params: { chatId: newChatId }, search: { isNew: true } });
      },
      [navigationRequestTypes.openPreferences]: () => navigate({ to: "/settings", search: { tab: "general" } }),
    };

    const offPlatform = getPlatform().system.onNavigationRequest((req) => {
      const handler = handlers[req.type] as (r: NavigationRequest) => void;
      handler(req);
    });

    const onOpenChatForModelDom = (ev: Event) => {
      if (!(ev instanceof CustomEvent)) return;
      const identifier = typeof ev.detail?.identifier === "string" ? ev.detail.identifier : "";
      if (!identifier) return;
      openNewChatForModel(identifier);
    };

    window.addEventListener(navigationRequestTypes.openChatForModel, onOpenChatForModelDom as EventListener);

    return () => {
      offPlatform();
      window.removeEventListener(navigationRequestTypes.openChatForModel, onOpenChatForModelDom as EventListener);
    };
  }, [createNewChat, navigate, openNewChatForModel]);
};

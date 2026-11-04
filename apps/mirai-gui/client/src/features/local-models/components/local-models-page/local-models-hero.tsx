import { UzuCat } from "@/components/uzu-cat";

/**
 * Above the catalog: the cat shows where these models live. Its brackets close into an iPhone,
 * widen into a Mac, and open back into its face, on a loop.
 */
export function LocalModelsHero() {
  return (
    <div className="flex flex-col items-center pb-8 text-center animate-uzu-in">
      <div className="mb-1">
        <UzuCat state="local" size={112} className="text-text-primary" label="uzu, becoming an iPhone and a Mac" />
      </div>
      <div className="flex flex-col gap-1">
        <h2 className="text-[15px] font-[500] leading-[1.35] text-label-title text-balance">
          Pick a local model to chat
        </h2>
        <p className="text-[13px] leading-[1.45] text-label-muted text-pretty">It runs right here, on your device</p>
      </div>
    </div>
  );
}

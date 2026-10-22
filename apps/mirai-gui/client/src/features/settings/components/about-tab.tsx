import { ChevronRight, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";
const links = [
  {
    title: "Website",
    url: "https://trymirai.com/",
  },
  { title: "GitHub", url: "https://github.com/trymirai" },
  { title: "Vision", url: "https://trymirai.com/about-us" },
  { title: "Docs", url: "https://docs.trymirai.com/" },
];

const productLinks = [
  {
    title: "Mirai platform",
    description: "A web console for setting up the SDK in your product.",
    url: "https://platform.trymirai.com",
  },
  {
    title: "Command-line interface",
    description: "Send messages to a model interactively, or start a local server.",
    url: "https://github.com/trymirai/uzu#cli",
  },
  {
    title: "Rust inference engine",
    description: "Runs models on the hardware they were built for.",
    url: "https://github.com/trymirai/uzu",
  },
];

const legalLinks = [
  {
    title: "Terms of Service",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Terms_of_Use.pdf",
  },
  {
    title: "Privacy Policy",
    url: "https://artifacts.trymirai.com/legal/Mirai_Tech_Privacy_Policy.pdf",
  },
];

export default function AboutTab() {
  return (
    <div className="h-full flex flex-col justify-between">
      <div className="flex flex-col">
        <div className="flex flex-col gap-4 px-5 lg:max-w-[800px] mx-auto w-full">
          <Heart className="w-6 h-6 text-blue" />
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h3 className="text-[18px] font-medium leading-[130%] tracking-[0.2px] text-label-title">
              Built by a team that shares a vision for <br />
              accessible and powerful local AI.
            </h3>
            <Button
              size="sm"
              href="https://discord.gg/trymirai"
              target="_blank"
              className="bg-blue/10 text-blue hover:bg-blue/20 active:bg-blue/25"
            >
              Join our Discord
            </Button>
          </div>
        </div>
        <div className="mt-5 h-[1px] bg-cell-border" />

        <div className="grid grid-cols-4 lg:max-w-[800px] mx-auto w-full">
          {links.map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:bg-bg-sub flex items-center justify-between px-5 py-5 border-r last:border-r-0 border-cell-border"
            >
              <h4 className="min-w-0 text-[15px] font-[350] leading-[150%] text-label-title overflow-clip">
                {link.title}
              </h4>
              <ChevronRight className="w-[14px] h-[14px] text-label-muted" />
            </a>
          ))}
        </div>

        <div className="mb-5 h-[1px] bg-cell-border" />
      </div>

      <div>
        <h3 className="lg:max-w-[800px] mx-auto w-full text-xs font-mono leading-[130%] text-label-muted px-5">
          Our products
        </h3>
        <div className="flex flex-col w-full mt-3">
          {productLinks.map((link, index) => (
            <div key={link.url} className="w-full">
              <a
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="lg:max-w-[800px] mx-auto w-full hover:bg-bg-sub flex items-center justify-between px-5 py-5"
              >
                <div className="flex flex-col gap-1">
                  <h4 className="text-[15px] font-[350] leading-[150%] text-label-title overflow-clip">{link.title}</h4>
                  <p className="text-[13px] font-[350] leading-[150%] text-label-muted">{link.description}</p>
                </div>

                <ChevronRight className="w-[14px] h-[14px] text-label-muted" />
              </a>
              {index < productLinks.length - 1 && <div className="h-[1px] bg-cell-border" />}
            </div>
          ))}
        </div>

        <div className="h-[1px] bg-cell-border" />
        <div className="grid grid-cols-2 lg:max-w-[800px] mx-auto w-full">
          {legalLinks.map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:bg-bg-sub flex items-center justify-between px-5 py-5 border-r last:border-r-0 border-cell-border"
            >
              <h4 className="text-[15px] font-[350] leading-[150%] text-label-title">{link.title}</h4>
              <ChevronRight className="w-[14px] h-[14px] text-label-muted" />
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

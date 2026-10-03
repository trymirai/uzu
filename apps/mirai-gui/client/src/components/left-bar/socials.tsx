import DiscordIcon from "../icons/discord-icon";
import GithubIcon from "../icons/github-icon";
import XIcon from "../icons/x-icon";

function Socials() {
  return (
    <div className="flex items-center justify-start space-x-4">
      <a
        href="https://github.com/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted hover:text-label-title"
      >
        <GithubIcon className="w-5 h-5" />
      </a>
      <a
        href="https://x.com/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted hover:text-label-title"
      >
        <XIcon className="w-4 h-4" />
      </a>
      <a
        href="https://discord.gg/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted hover:text-label-title"
      >
        <DiscordIcon className="w-5 h-5" />
      </a>
    </div>
  );
}

export default Socials;

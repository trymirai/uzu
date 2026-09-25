import DiscordIcon from "../icons/DiscordIcon";
import GithubIcon from "../icons/GithubIcon";
import XIcon from "../icons/XIcon";

function Socials() {
  return (
    <div className="flex items-center justify-start space-x-4">
      <a
        href="https://github.com/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted dark:text-label-muted-dark hover:text-label-title dark:hover:text-label-title-dark"
      >
        <GithubIcon className="w-5 h-5" />
      </a>
      <a
        href="https://x.com/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted dark:text-label-muted-dark hover:text-label-title dark:hover:text-label-title-dark"
      >
        <XIcon className="w-4 h-4" />
      </a>
      <a
        href="https://discord.gg/trymirai"
        target="_blank"
        rel="noopener noreferrer"
        className="text-label-muted dark:text-label-muted-dark hover:text-label-title dark:hover:text-label-title-dark"
      >
        <DiscordIcon className="w-5 h-5" />
      </a>
    </div>
  );
}

export default Socials;

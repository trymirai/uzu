import React from "react";

type CodeIconProps = {
  className?: string;
};

export const CodeIcon: React.FC<CodeIconProps> = ({ className }) => {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <path
        d="M4.11111 4.88845L1 7.99957L4.11111 11.1107M11.8889 4.88845L15 7.99957L11.8889 11.1107M9.55556 1.77734L6.44444 14.2218"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

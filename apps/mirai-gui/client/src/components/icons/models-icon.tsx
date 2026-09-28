import React from "react";

type ModelsIconProps = React.SVGProps<SVGSVGElement>;

export function ModelsIcon(props: ModelsIconProps) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="none" {...props}>
      <path
        d="M6.41667 7.33366L3.75 10.0003L6.41667 12.667M13.0833 7.33366L15.75 10.0003L13.0833 12.667M11.0833 4.66699L8.41667 15.3337"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

import React from "react";

const SidebarToggleIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20" fill="none" {...props}>
    <path
      d="M7.75 4V16M4 5.5C4 5.10218 4.15804 4.72064 4.43934 4.43934C4.72064 4.15804 5.10218 4 5.5 4H14.5C14.8978 4 15.2794 4.15804 15.5607 4.43934C15.842 4.72064 16 5.10218 16 5.5V14.5C16 14.8978 15.842 15.2794 15.5607 15.5607C15.2794 15.842 14.8978 16 14.5 16H5.5C5.10218 16 4.72064 15.842 4.43934 15.5607C4.15804 15.2794 4 14.8978 4 14.5V5.5Z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export default SidebarToggleIcon;

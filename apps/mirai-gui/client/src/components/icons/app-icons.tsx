import { type SVGProps, useId } from "react";

/*
 * The interface icons, as local SVG components (24px grid, 2px round strokes). Each takes `size`
 * (16, 20, ...) and any SVG prop; they are decorative by default (`aria-hidden`), so a button
 * names itself with `aria-label`. Colour comes from `currentColor`.
 */

export type IconProps = Omit<SVGProps<SVGSVGElement>, "children"> & { size?: number | string };

function Svg({ size = 24, children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      aria-hidden
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...rest}
    >
      {children}
    </svg>
  );
}

export function CircleArrowDownIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12C21 16.9706 16.9706 21 12 21Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 8V16M12 16L15 13M12 16L9 13"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/** Regenerate: a clockwise arrow round a circle (the Figma regenerate icon, scaled from its 32px grid). */
export function RotateCwIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M21 12C21 13.78 20.4721 15.5201 19.4832 17.0001C18.4943 18.4802 17.0887 19.6337 15.4441 20.3149C13.7996 20.9961 11.99 21.1743 10.2442 20.827C8.4984 20.4798 6.8947 19.6226 5.636 18.364C4.3774 17.1053 3.5202 15.5017 3.1729 13.7558C2.8257 12.01 3.0039 10.2004 3.6851 8.5558C4.3663 6.9113 5.5198 5.5057 6.9999 4.5168C8.4799 3.5278 10.22 3 12 3C14.52 3 16.93 4 18.74 5.74L21 8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M21 3V8H16.0001" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function InfoIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M11 11H12V16M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 7.25C11.5858 7.25 11.25 7.58579 11.25 8C11.25 8.41421 11.5858 8.75 12 8.75C12.4142 8.75 12.75 8.41421 12.75 8C12.75 7.58579 12.4142 7.25 12 7.25Z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="0.5"
      />
    </Svg>
  );
}

export function XIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M8 8L16 16M16 8L8 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </Svg>
  );
}

export function EllipsisIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M20 13C20.5523 13 21 12.5523 21 12C21 11.4477 20.5523 11 20 11C19.4477 11 19 11.4477 19 12C19 12.5523 19.4477 13 20 13Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 13C4.55228 13 5 12.5523 5 12C5 11.4477 4.55228 11 4 11C3.44772 11 3 11.4477 3 12C3 12.5523 3.44772 13 4 13Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/** Alert: an exclamation mark in a circle (the Figma alert icon, scaled from its 32px grid). Used
 * for every alert, warnings and errors alike; the colour tells them apart. */
export function CircleAlertIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M12 22C17.5229 22 22.0001 17.5229 22.0001 12C22.0001 6.4771 17.5229 2 12 2C6.4772 2 2 6.4771 2 12C2 17.5229 6.4772 22 12 22Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M12 8V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 16H12.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

/** A lightning bolt, for speed (the Figma speed icon). */
export function ZapIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M15.914 4C16.0271 3.67936 16.028 3.3298 15.9164 3.00861C15.8049 2.68743 15.5875 2.41367 15.2999 2.23224C15.0124 2.0508 14.6717 1.97244 14.3338 2.01001C13.9958 2.04757 13.6807 2.19884 13.44 2.439L4.43999 11.439C4.23013 11.6487 4.08718 11.916 4.02921 12.2069C3.97125 12.4979 4.00087 12.7995 4.11433 13.0737C4.2278 13.3478 4.42001 13.5821 4.66665 13.747C4.9133 13.9119 5.2033 13.9999 5.49999 14H9.50199C9.58186 14.0001 9.66055 14.0193 9.73147 14.0561C9.80238 14.0929 9.86345 14.1461 9.90958 14.2113C9.9557 14.2765 9.98552 14.3518 9.99655 14.4309C10.0076 14.5101 9.9995 14.5907 9.97299 14.666L8.08599 20C7.97279 20.3207 7.97198 20.6704 8.08366 20.9917C8.19535 21.313 8.41291 21.5868 8.70066 21.7681C8.9884 21.9495 9.32925 22.0277 9.66727 21.9899C10.0053 21.9521 10.3204 21.8005 10.561 21.56L19.561 12.56C19.7706 12.3502 19.9132 12.0829 19.971 11.792C20.0287 11.5011 19.9989 11.1997 19.8854 10.9257C19.7718 10.6517 19.5796 10.4176 19.333 10.2528C19.0864 10.088 18.7966 10.0001 18.5 10H14.503C14.4229 10.0001 14.344 9.98099 14.2728 9.94424C14.2017 9.90749 14.1404 9.85419 14.0941 9.78882C14.0479 9.72345 14.018 9.64793 14.007 9.56861C13.996 9.48928 14.0042 9.40848 14.031 9.333L15.914 4Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function FileTextIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M5 6C5 4.34315 6.34315 3 8 3H16C17.6569 3 19 4.34315 19 6V18C19 19.6569 17.6569 21 16 21H8C6.34315 21 5 19.6569 5 18V6Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M9 7H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9 11H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M9 15H11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

/** Rename: a rounded pencil (the Figma rename icon, scaled from its 20px grid), the same pencil as
 * the sidebar's New chat without the line under it. */
export function PencilIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M20.2213 4.1213L20.4787 4.3787C21.6503 5.5503 21.6503 7.4498 20.4787 8.6214L8.9787 20.1214C8.4161 20.684 7.653 21 6.8574 21H3.6V17.7427C3.6 16.947 3.9161 16.184 4.4787 15.6214L15.9787 4.1214C17.1503 2.9498 19.0496 2.9498 20.2213 4.1213Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M12 5.5V12M12 12V18.5M12 12H5.5M12 12H18.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </Svg>
  );
}

/** Copy: a square over the corner of the one behind it (the Figma copy icon, scaled from its 32px grid). */
export function CopyIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M20 8H10C8.8955 8 8 8.8955 8 10V20C8 21.1045 8.8955 22 10 22H20C21.1045 22 22.0001 21.1045 22.0001 20V10C22.0001 8.8955 21.1045 8 20 8Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 16C2.9 16 2 15.1 2 14V4C2 2.9 2.9 2 4 2H14C15.1001 2 16.0001 2.9 16.0001 4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/** Download: an arrow dropping into a tray. */
export function DownloadIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M4 14C4.55228 14 5 14.4477 5 15V17C5 18.1046 5.89543 19 7 19H17C18.1046 19 19 18.1046 19 17V15C19 14.4477 19.4477 14 20 14C20.5523 14 21 14.4477 21 15V17C21 19.2091 19.2091 21 17 21H7C4.79086 21 3 19.2091 3 17V15C3 14.4477 3.44772 14 4 14Z"
        fillRule="evenodd"
        clipRule="evenodd"
        fill="currentColor"
      />
      <path
        d="M12 15.5C12.2652 15.5 12.5196 15.3946 12.7071 15.2071L16.2071 11.7071C16.5976 11.3166 16.5976 10.6834 16.2071 10.2929C15.8166 9.90237 15.1834 9.90237 14.7929 10.2929L13 12.0858V4C13 3.44771 12.5523 3 12 3C11.4477 3 11 3.44771 11 4V12.0858L9.20711 10.2929C8.81658 9.90237 8.18342 9.90237 7.79289 10.2929C7.40237 10.6834 7.40237 11.3166 7.79289 11.7071L11.2929 15.2071C11.4804 15.3946 11.7348 15.5 12 15.5Z"
        fillRule="evenodd"
        clipRule="evenodd"
        fill="currentColor"
      />
    </Svg>
  );
}

/** The circle-check, filled: a solid disc with the check cut out of it. Shown after a copy. */
export function CircleCheckIcon({ size = 16, ...rest }: IconProps) {
  const maskId = useId();
  return (
    <Svg size={size} {...rest}>
      <mask id={maskId} maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">
        <rect width="24" height="24" fill="#fff" />
        <path d="M15 9.5L10.5 15L8.5 13" stroke="#000" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
      </mask>
      <circle cx="12" cy="12" r="10" fill="currentColor" mask={`url(#${maskId})`} />
    </Svg>
  );
}

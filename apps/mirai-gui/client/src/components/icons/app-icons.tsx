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

export function IconArrowDownCircle(props: IconProps) {
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

export function IconArrowRotateClockwise(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M19.5165 14.6667C18.4182 17.7738 15.4549 20 11.9717 20C7.5534 20 3.97168 16.4183 3.97168 12C3.97168 7.58172 7.5534 4 11.9717 4C15.0676 4 17 5.5 19 8"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M19 4V8H15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconCircleCheck(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M15 9.5L10.5 15L8.5 13M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function IconCircleInfo(props: IconProps) {
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

export function IconCrossSmall(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M8 8L16 16M16 8L8 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </Svg>
  );
}

export function IconDotGrid1x3Horizontal(props: IconProps) {
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

export function IconExclamationCircle(props: IconProps) {
  return (
    <Svg {...props}>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
      <path d="M12 8V12.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <circle cx="12" cy="15.7996" r="1.2" fill="currentColor" />
    </Svg>
  );
}

export function IconExclamationTriangle(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M12 8.8846V11.9192M12 15H12.01M9.48223 4.43254L3.45153 14.3633C2.22395 16.3848 3.64405 19 5.9693 19H18.0307C20.3559 19 21.7761 16.3848 20.5485 14.3633L14.5178 4.43254C13.3578 2.52249 10.6422 2.52249 9.48223 4.43254ZM12.25 15C12.25 15.1381 12.1381 15.25 12 15.25C11.8619 15.25 11.75 15.1381 11.75 15C11.75 14.8619 11.8619 14.75 12 14.75C12.1381 14.75 12.25 14.8619 12.25 15Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </Svg>
  );
}

export function IconGauge(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M10 10L5.75 5.75M12 5V3C16.9706 3 21 7.02944 21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 10.9481 3.18046 9.93834 3.51212 9M14.5 12C14.5 13.3807 13.3807 14.5 12 14.5C10.6193 14.5 9.5 13.3807 9.5 12C9.5 10.6193 10.6193 9.5 12 9.5C13.3807 9.5 14.5 10.6193 14.5 12Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function IconPageText(props: IconProps) {
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

export function IconPencil(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M13.5 5.99997L15.25 4.24997C16.4926 3.00733 18.5074 3.00733 19.75 4.24997C20.9926 5.49261 20.9926 7.50733 19.75 8.74997L18 10.5L7.79289 20.7071C7.60536 20.8946 7.351 21 7.08579 21H3V16.9142C3 16.649 3.10536 16.3946 3.29289 16.2071L13.5 5.99997Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M13.5 6L18 10.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconPlusMedium(props: IconProps) {
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

export function IconSettingsGear2(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M7.67668 5.32539L7.30515 5.23965C6.72652 5.10612 6.11991 5.28009 5.7 5.7C5.28009 6.11991 5.10612 6.72651 5.23965 7.30515L5.32539 7.67668C5.55082 8.65357 5.148 9.668 4.31382 10.2241L3.74885 10.6008C3.28101 10.9127 3 11.4377 3 12C3 12.5623 3.28101 13.0873 3.74885 13.3992L4.31382 13.7759C5.148 14.332 5.55082 15.3464 5.32539 16.3233L5.23965 16.6949C5.10612 17.2735 5.28009 17.8801 5.7 18.3C6.11991 18.7199 6.72651 18.8939 7.30515 18.7604L7.67668 18.6746C8.65357 18.4492 9.668 18.852 10.2241 19.6862L10.6008 20.2512C10.9127 20.719 11.4377 21 12 21C12.5623 21 13.0873 20.719 13.3992 20.2512L13.7759 19.6862C14.332 18.852 15.3464 18.4492 16.3233 18.6746L16.6949 18.7604C17.2735 18.8939 17.8801 18.7199 18.3 18.3C18.7199 17.8801 18.8939 17.2735 18.7604 16.6949L18.6746 16.3233C18.4492 15.3464 18.852 14.332 19.6862 13.7759L20.2512 13.3992C20.719 13.0873 21 12.5623 21 12C21 11.4377 20.719 10.9127 20.2512 10.6008L19.6862 10.2241C18.852 9.668 18.4492 8.65357 18.6746 7.67668L18.7604 7.30515C18.8939 6.72652 18.7199 6.11991 18.3 5.7C17.8801 5.28009 17.2735 5.10612 16.6949 5.23965L16.3233 5.32539C15.3464 5.55082 14.332 5.148 13.7759 4.31382L13.3992 3.74884C13.0873 3.28101 12.5623 3 12 3C11.4377 3 10.9127 3.28101 10.6008 3.74885L10.2241 4.31382C9.668 5.148 8.65357 5.55082 7.67668 5.32539Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function IconSquareBehindSquare1(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M15 9V5.25C15 4.00736 13.9926 3 12.75 3H5.25C4.00736 3 3 4.00736 3 5.25V12.75C3 13.9926 4.00736 15 5.25 15H9M11.25 9H18.75C19.9926 9 21 10.0074 21 11.25V18.75C21 19.9926 19.9926 21 18.75 21H11.25C10.0074 21 9 19.9926 9 18.75V11.25C9 10.0074 10.0074 9 11.25 9Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function IconTrashCan(props: IconProps) {
  return (
    <Svg {...props}>
      <path
        d="M5 6.5L5.80734 18.2064C5.91582 19.7794 7.22348 21 8.80023 21H15.1998C16.7765 21 18.0842 19.7794 18.1927 18.2064L19 6.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M10 11V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 11V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3.5 6H20.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path
        d="M8.07092 5.74621C8.42348 3.89745 10.0485 2.5 12 2.5C13.9515 2.5 15.5765 3.89745 15.9291 5.74621"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/** Download: an arrow dropping into a tray. */
export function IconArrowInbox(props: IconProps) {
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
export function IconCircleCheckFilled({ size = 16, ...rest }: IconProps) {
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

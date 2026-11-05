import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";

/*
 * The sidebar's icons, drawn like the rest of the app's icons (components/icons/app-icons: 24px
 * grid, 2px round strokes) and split into parts so one part can move when the row is hovered. Each motion is a small nudge that
 * says what the item does, on a spring so leaving halfway simply turns back. Pointer hover only:
 * keyboard focus doesn't animate, and reduced motion keeps every icon still.
 */

type NavIconProps = { size?: number; className?: string; active?: boolean };

// Quick and nearly flat: hover feedback, not a show. The bubble gets a touch more bounce.
const SPRING = { type: "spring", visualDuration: 0.3, bounce: 0.2 } as const;
const WOBBLE = { type: "spring", visualDuration: 0.32, bounce: 0.35 } as const;
const TURN = { type: "spring", visualDuration: 0.45, bounce: 0.15 } as const;

function Svg({ size = 20, className, children }: { size?: number; className?: string; children: React.ReactNode }) {
  return (
    <svg
      aria-hidden
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      className={className}
      style={{ overflow: "visible" }}
    >
      {children}
    </svg>
  );
}

/** Models: still. It takes `active` like the others so the rows can treat every icon alike. */
export function ModelsIcon({ size, className }: NavIconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M9.6001 3.38565C10.4744 2.88086 10.9116 2.62846 11.3764 2.52967C11.7876 2.44226 12.2126 2.44226 12.6238 2.52967C13.0886 2.62846 13.5258 2.88086 14.4001 3.38565L18.2604 5.61437C19.1347 6.11916 19.5718 6.37156 19.8898 6.72467C20.1711 7.0371 20.3836 7.40517 20.5135 7.80501C20.6604 8.25691 20.6604 8.7617 20.6604 9.77129V14.2287C20.6604 15.2383 20.6604 15.7431 20.5135 16.195C20.3836 16.5949 20.1711 16.9629 19.8898 17.2754C19.5718 17.6285 19.1347 17.8809 18.2604 18.3857L14.4001 20.6144C13.5258 21.1192 13.0886 21.3716 12.6238 21.4704C12.2126 21.5578 11.7876 21.5578 11.3764 21.4704C10.9116 21.3716 10.4744 21.1192 9.6001 20.6144L5.73984 18.3857C4.86551 17.8809 4.42835 17.6285 4.11041 17.2754C3.8291 16.9629 3.61659 16.5949 3.48667 16.195C3.33984 15.7431 3.33984 15.2383 3.33984 14.2287V9.77129C3.33984 8.7617 3.33984 8.25691 3.48667 7.80501C3.61659 7.40517 3.8291 7.0371 4.11041 6.72467C4.42835 6.37156 4.86551 6.11916 5.73984 5.61437L9.6001 3.38565Z" />
      <path d="M20.6604 7L12 12M12 22V12M3.33984 7L12 12" />
    </Svg>
  );
}

/** Chats: the bubble tips up from its tail, as if it just popped in. */
export function ChatsIcon({ size, className, active = false }: NavIconProps) {
  const still = useReducedMotion();
  const on = active && !still;
  return (
    <Svg size={size} className={className}>
      <motion.path
        d="M21.5 12C21.5 7 17.8056 4 12 4C6.19444 4 2.5 7 2.5 12C2.5 13.2943 3.39422 15.4896 3.53656 15.8309C3.54957 15.862 3.56246 15.8905 3.57409 15.9222C3.67156 16.188 4.06312 17.5822 2.5 19.6439C4.61111 20.6439 6.8531 19 6.8531 19C8.40425 19.8154 10.2499 20 12 20C17.8056 20 21.5 17 21.5 12Z"
        strokeLinecap="square"
        strokeLinejoin="round"
        style={{ originX: 0.05, originY: 0.95 }}
        initial={false}
        animate={{ rotate: on ? -7 : 0, scale: on ? 1.04 : 1 }}
        transition={WOBBLE}
      />
    </Svg>
  );
}

/** New chat: the pencil tilts on its tip, as if to start writing. */
export function NewChatIcon({ size, className, active = false }: NavIconProps) {
  const still = useReducedMotion();
  const on = active && !still;
  return (
    <Svg size={size} className={className}>
      <motion.path
        d="M19.6213 4.12135L19.8787 4.37872C21.0503 5.55029 21.0503 7.44978 19.8787 8.62136L8.37868 20.1214C7.81607 20.684 7.05301 21 6.25736 21H3V17.7427C3 16.947 3.31607 16.184 3.87868 15.6214L15.3787 4.12136C16.5503 2.94978 18.4497 2.94978 19.6213 4.12135Z"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ originX: 0, originY: 1 }}
        initial={false}
        animate={{ rotate: on ? -9 : 0 }}
        transition={SPRING}
      />
      <path d="M14 21H21" strokeLinecap="round" />
    </Svg>
  );
}

/**
 * Settings: the gear turns one notch. It has eight teeth, so 45° lands on the same shape; each
 * hover turns it further the same way, like a dial, rather than winding back.
 */
export function SettingsIcon({ size, className, active = false }: NavIconProps) {
  const still = useReducedMotion();
  const [turns, setTurns] = useState(0);
  const wasActive = useRef(active);
  useEffect(() => {
    if (active && !wasActive.current && !still) setTurns((n) => n + 1);
    wasActive.current = active;
  }, [active, still]);
  return (
    <Svg size={size} className={className}>
      <motion.g
        style={{ originX: 0.5, originY: 0.5 }}
        initial={false}
        animate={{ rotate: turns * 45 }}
        transition={TURN}
      >
        <path
          d="M7.67668 5.32539L7.30515 5.23965C6.72652 5.10612 6.11991 5.28009 5.7 5.7C5.28009 6.11991 5.10612 6.72651 5.23965 7.30515L5.32539 7.67668C5.55082 8.65357 5.148 9.668 4.31382 10.2241L3.74885 10.6008C3.28101 10.9127 3 11.4377 3 12C3 12.5623 3.28101 13.0873 3.74885 13.3992L4.31382 13.7759C5.148 14.332 5.55082 15.3464 5.32539 16.3233L5.23965 16.6949C5.10612 17.2735 5.28009 17.8801 5.7 18.3C6.11991 18.7199 6.72651 18.8939 7.30515 18.7604L7.67668 18.6746C8.65357 18.4492 9.668 18.852 10.2241 19.6862L10.6008 20.2512C10.9127 20.719 11.4377 21 12 21C12.5623 21 13.0873 20.719 13.3992 20.2512L13.7759 19.6862C14.332 18.852 15.3464 18.4492 16.3233 18.6746L16.6949 18.7604C17.2735 18.8939 17.8801 18.7199 18.3 18.3C18.7199 17.8801 18.8939 17.2735 18.7604 16.6949L18.6746 16.3233C18.4492 15.3464 18.852 14.332 19.6862 13.7759L20.2512 13.3992C20.719 13.0873 21 12.5623 21 12C21 11.4377 20.719 10.9127 20.2512 10.6008L19.6862 10.2241C18.852 9.668 18.4492 8.65357 18.6746 7.67668L18.7604 7.30515C18.8939 6.72652 18.7199 6.11991 18.3 5.7C17.8801 5.28009 17.2735 5.10612 16.6949 5.23965L16.3233 5.32539C15.3464 5.55082 14.332 5.148 13.7759 4.31382L13.3992 3.74884C13.0873 3.28101 12.5623 3 12 3C11.4377 3 10.9127 3.28101 10.6008 3.74885L10.2241 4.31382C9.668 5.148 8.65357 5.55082 7.67668 5.32539Z"
          strokeLinejoin="round"
        />
        <path
          d="M15 12C15 13.6569 13.6569 15 12 15C10.3431 15 9 13.6569 9 12C9 10.3431 10.3431 9 12 9C13.6569 9 15 10.3431 15 12Z"
          strokeLinejoin="round"
        />
      </motion.g>
    </Svg>
  );
}

/**
 * Sidebar toggle. The divider shows where the sidebar edge is: at rest it's there when the sidebar
 * is open and gone when it's closed. Hovering previews the click: an open sidebar's divider slides
 * fully into the left edge and fades, a closed one's slides back out to where the sidebar would be.
 * The caller freezes `active` after a click, so the icon keeps its look until the pointer leaves.
 * Under reduced motion the state still changes, it just doesn't slide.
 */
export function SidebarIcon({ size, className, active = false, open }: NavIconProps & { open: boolean }) {
  const still = useReducedMotion();
  const shown = open !== active;
  const x = shown ? 9 : 3;
  return (
    <Svg size={size} className={className}>
      <path
        d="M3 8C3 6.34315 4.34315 5 6 5H18C19.6569 5 21 6.34315 21 8V16C21 17.6569 19.6569 19 18 19H6C4.34315 19 3 17.6569 3 16V8Z"
        strokeLinejoin="round"
      />
      <motion.path
        strokeLinejoin="round"
        initial={false}
        animate={{ d: `M${x} 5V19`, opacity: shown ? 1 : 0 }}
        transition={
          still
            ? { duration: 0 }
            : {
                d: SPRING,
                // Gone by the time it reaches the edge, back in as soon as it leaves it.
                opacity: shown ? { duration: 0.12, ease: "easeOut" } : { duration: 0.2, ease: "easeIn", delay: 0.06 },
              }
        }
      />
    </Svg>
  );
}

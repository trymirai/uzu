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

/** Models: a rounded square that turns a little about its centre, on the same spring and by the same
 * angle as the New chat pencil. */
export function ModelsIcon({ size, className, active = false }: NavIconProps) {
  const still = useReducedMotion();
  const on = active && !still;
  return (
    <Svg size={size} className={className}>
      <motion.path
        d="M17 4H7C5.34315 4 4 5.34315 4 7V17C4 18.6569 5.34315 20 7 20H17C18.6569 20 20 18.6569 20 17V7C20 5.34315 18.6569 4 17 4Z"
        strokeLinejoin="round"
        style={{ originX: 0.5, originY: 0.5 }}
        initial={false}
        animate={{ rotate: on ? -9 : 0 }}
        transition={SPRING}
      />
    </Svg>
  );
}

/** Chats: a round bubble with a square corner for its tail (the Figma "Icons / Chat" shape, drawn
 * as a filled outline), which tips up from that corner, as if it just popped in. */
export function ChatsIcon({ size, className, active = false }: NavIconProps) {
  const still = useReducedMotion();
  const on = active && !still;
  return (
    <Svg size={size} className={className}>
      <motion.path
        d="M12.375 1.875C9.78998 1.87773 7.31161 2.90584 5.48372 4.73372C3.65584 6.56161 2.62773 9.03998 2.625 11.625V19.5C2.625 19.9973 2.82254 20.4742 3.17417 20.8258C3.52581 21.1775 4.00272 21.375 4.5 21.375H12.375C14.9609 21.375 17.4408 20.3478 19.2693 18.5193C21.0978 16.6908 22.125 14.2109 22.125 11.625C22.125 9.03914 21.0978 6.55919 19.2693 4.73071C17.4408 2.90223 14.9609 1.875 12.375 1.875ZM12.375 19.125H4.875V11.625C4.875 10.1416 5.31487 8.69159 6.13898 7.45822C6.96309 6.22485 8.13443 5.26356 9.50487 4.6959C10.8753 4.12825 12.3833 3.97972 13.8382 4.26911C15.293 4.5585 16.6294 5.27281 17.6783 6.3217C18.7272 7.37059 19.4415 8.70697 19.7309 10.1618C20.0203 11.6167 19.8718 13.1247 19.3041 14.4951C18.7364 15.8656 17.7751 17.0369 16.5418 17.861C15.3084 18.6851 13.8584 19.125 12.375 19.125Z"
        fill="currentColor"
        stroke="none"
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
 * Settings: a rounded six-lobed gear (the Figma settings icon) that turns one notch. It has six
 * lobes, so 60° lands on the same shape; each hover turns it further the same way, like a dial,
 * rather than winding back.
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
        animate={{ rotate: turns * 60 }}
        transition={TURN}
      >
        <path
          d="M9.671 4.136C9.7261 3.55634 9.99533 3.01804 10.4261 2.62628C10.8569 2.23451 11.4182 2.01743 12.0005 2.01743C12.5828 2.01743 13.1441 2.23451 13.5749 2.62628C14.0057 3.01804 14.2749 3.55634 14.33 4.136C14.3631 4.51045 14.486 4.87142 14.6881 5.18834C14.8903 5.50526 15.1659 5.76881 15.4915 5.95668C15.8171 6.14454 16.1831 6.2512 16.5587 6.26762C16.9343 6.28404 17.3082 6.20973 17.649 6.051C18.1781 5.81078 18.7777 5.77602 19.331 5.95348C19.8843 6.13095 20.3518 6.50795 20.6425 7.0111C20.9332 7.51426 21.0263 8.10757 20.9036 8.67557C20.781 9.24357 20.4514 9.74561 19.979 10.084C19.6714 10.2998 19.4203 10.5866 19.2469 10.92C19.0736 11.2534 18.983 11.6237 18.983 11.9995C18.983 12.3753 19.0736 12.7456 19.2469 13.079C19.4203 13.4124 19.6714 13.6992 19.979 13.915C20.4514 14.2534 20.781 14.7554 20.9036 15.3234C21.0263 15.8914 20.9332 16.4847 20.6425 16.9879C20.3518 17.491 19.8843 17.868 19.331 18.0455C18.7777 18.223 18.1781 18.1882 17.649 17.948C17.3082 17.7893 16.9343 17.715 16.5587 17.7314C16.1831 17.7478 15.8171 17.8545 15.4915 18.0423C15.1659 18.2302 14.8903 18.4937 14.6881 18.8107C14.486 19.1276 14.3631 19.4885 14.33 19.863C14.2749 20.4427 14.0057 20.981 13.5749 21.3727C13.1441 21.7645 12.5828 21.9816 12.0005 21.9816C11.4182 21.9816 10.8569 21.7645 10.4261 21.3727C9.99533 20.981 9.7261 20.4427 9.671 19.863C9.63794 19.4884 9.5151 19.1273 9.31286 18.8103C9.11063 18.4932 8.83497 18.2296 8.50923 18.0417C8.18349 17.8538 7.81727 17.7472 7.44158 17.7309C7.06589 17.7146 6.6918 17.7891 6.351 17.948C5.82189 18.1882 5.22233 18.223 4.669 18.0455C4.11567 17.868 3.64817 17.491 3.35748 16.9879C3.06679 16.4847 2.97371 15.8914 3.09636 15.3234C3.219 14.7554 3.5486 14.2534 4.021 13.915C4.32862 13.6992 4.57973 13.4124 4.75309 13.079C4.92645 12.7456 5.01695 12.3753 5.01695 11.9995C5.01695 11.6237 4.92645 11.2534 4.75309 10.92C4.57973 10.5866 4.32862 10.2998 4.021 10.084C3.54926 9.74544 3.22025 9.24359 3.0979 8.67598C2.97555 8.10837 3.06861 7.51554 3.35898 7.01271C3.64936 6.50988 4.11631 6.13298 4.66909 5.95524C5.22187 5.7775 5.82098 5.81163 6.35 6.051C6.69076 6.20973 7.06474 6.28404 7.4403 6.26762C7.81586 6.2512 8.18193 6.14454 8.50754 5.95668C8.83314 5.76881 9.10869 5.50526 9.31086 5.18834C9.51304 4.87142 9.63588 4.51045 9.669 4.136"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </motion.g>
    </Svg>
  );
}

/**
 * Sidebar toggle. The divider shows where the sidebar edge is: there when the sidebar is open, slid
 * into the left edge and faded when it's closed. It moves only when the sidebar opens or closes,
 * never on hover. Under reduced motion it changes without sliding.
 */
export function SidebarIcon({ size, className, open }: Omit<NavIconProps, "active"> & { open: boolean }) {
  const still = useReducedMotion();
  const shown = open;
  // Open, the divider stands a third of the way in; closed, it slides into the frame's left edge.
  const x = shown ? 9 : 3.36;
  return (
    <Svg size={size} className={className}>
      {/* The Figma sidebar icon, scaled from its 20px grid; its frame is drawn a touch lighter than
          the divider, as there. */}
      <rect x="3.36" y="5.16" width="17.28" height="13.68" rx="2.64" strokeWidth={1.92} />
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

import { useId, type Ref } from "react";
import { Input, type InputProps } from "../Input";

export type TextFieldProps = {
  error?: string;
  // aria-describedby is owned here: it wires the control to the error message.
  controlProps: Omit<InputProps, "aria-describedby"> & { ref?: Ref<HTMLInputElement> };
};

export function TextField({ error, controlProps }: TextFieldProps) {
  const reactId = useId();
  const controlId = controlProps.id ?? `field-${reactId}`;
  const errorId = error ? `${controlId}-error` : undefined;

  return (
    <div className="flex flex-col w-full">
      <Input
        {...controlProps}
        id={controlId}
        kind={error ? "error" : (controlProps.kind ?? "default")}
        aria-describedby={errorId}
      />
      {errorId && (
        <p id={errorId} className="mt-1.5 text-xs text-red-500">
          {error}
        </p>
      )}
    </div>
  );
}

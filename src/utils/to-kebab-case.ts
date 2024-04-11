const toKebabCase = (str: string = ""): string =>
  str
    .match(/[\uAC00-\uD7AF]+|[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+/g)
    ?.map((word) => word.toLowerCase())
    .join("-") || "";

export default toKebabCase;
